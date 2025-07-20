import torch
from invokeai.invocation_api import (
    BaseInvocation,
    InputField,
    InvocationContext,
    invocation,
    OutputField,
    invocation_output,
    BaseInvocationOutput,
)
from invokeai.app.invocations.fields import FluxConditioningField
from invokeai.app.invocations.primitives import FluxConditioningOutput
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    FLUXConditioningInfo,
    ConditioningFieldData,
)
from invokeai.backend.util.logging import info, warning, error
import math

# Define a custom output class for control of naming/tooltip.
@invocation_output("flux_conditioning_blend_output")
class FluxConditioningBlendOutput(BaseInvocationOutput):
    """Output for the blended Flux Conditionings."""
    conditioning: FluxConditioningField = OutputField(description="The interpolated Flux Conditioning")


# SLERP function for torch tensors
def slerp(
    vec0: torch.Tensor,
    vec1: torch.Tensor,
    alpha: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Performs spherical linear interpolation (SLERP) between two *normalized* tensors.
    This function is designed to interpolate the *direction* of vectors.
    It expects input vectors to be normalized or handles normalization internally for dot product,
    but the output will be a unit vector if inputs are unit vectors.
    """
    # Ensure alpha is clamped to [0, 1] for valid interpolation
    if not (0.0 <= alpha <= 1.0):
        warning("Alpha value should be between 0 and 1. Clamping to [0, 1].")
        alpha = max(0.0, min(1.0, alpha))

    # Ensure tensors are float32 for calculations
    vec0_f = vec0.to(torch.float32)
    vec1_f = vec1.to(torch.float32)

    # Calculate the dot product, which is cos(theta) for normalized vectors
    dot_product = torch.sum(vec0_f * vec1_f)

    # Clamp dot product to handle potential floating point inaccuracies that might push it outside [-1, 1]
    dot_product = torch.clamp(dot_product, -1.0 + epsilon, 1.0 - epsilon)

    # Calculate the angle between the two vectors
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)

    # Handle nearly collinear vectors (angle close to 0 or pi)
    # If sin_omega is very small, vectors are nearly collinear or anti-collinear.
    # In such cases, SLERP becomes unstable, so we fall back to LERP.
    if sin_omega.item() < epsilon:
        info("Vectors are nearly collinear, performing linear interpolation (LERP) for direction instead of SLERP.")
        return (1.0 - alpha) * vec0_f + alpha * vec1_f

    # Apply the SLERP formula
    s0 = torch.sin((1.0 - alpha) * omega) / sin_omega
    s1 = torch.sin(alpha * omega) / sin_omega

    # Return the interpolated vector, casting back to the original dtype
    return (s0 * vec0_f + s1 * vec1_f).to(vec0.dtype)


@invocation(
    "flux_conditioning_blend",  # Unique identifier for the node
    title="Flux Conditioning Blend",  # Human-readable title for the UI
    tags=["conditioning", "flux", "interpolation", "slerp", "lerp", "blend"],  # Searchable tags
    category="conditioning",  # Category for grouping in the UI
    version="1.0.0",  # Increment version on changes
)
class FluxConditioningBlendInvocation(BaseInvocation):
    """
    Performs a blend between two FLUX Conditioning objects using either direct SLERP
    or an advanced method that separates magnitude and direction.
    """

    conditioning_1: FluxConditioningField = InputField(
        description="The first FLUX Conditioning object.", ui_order=0,
    )
    conditioning_2: FluxConditioningField = InputField(
        description="The second FLUX Conditioning object.", ui_order=1,
    )
    alpha: float = InputField(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Interpolation factor (0.0 for conditioning_1, 1.0 for conditioning_2).",
        ui_order=2,
    )
    use_magnitude_separation: bool = InputField(
        default=False,
        description="If True, uses magnitude separation (SLERP for direction, LERP for magnitude); otherwise, uses direct SLERP.",
        ui_order=3,
    )

    def _load_conditioning_info(
        self, context: InvocationContext, field: FluxConditioningField
    ) -> FLUXConditioningInfo | None:
        """Helper method to safely load FLUXConditioningInfo from a FluxConditioningField."""
        if field is None or field.conditioning_name is None:
            return None
        try:
            # context.conditioning.load returns a ConditioningFieldData object,
            # which contains a list of FLUXConditioningInfo objects.
            conditioning_data = context.conditioning.load(field.conditioning_name)
            if conditioning_data and conditioning_data.conditionings:
                # Assuming FLUXConditioningInfo is always the first element in the list
                return conditioning_data.conditionings[0]
            else:
                warning(f"No conditioning data found for name: {field.conditioning_name}")
                return None
        except Exception as e:
            error(f"Failed to load conditioning data for {field.conditioning_name}: {e}")
            return None

    def _normalize_and_get_magnitude(self, embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalizes the embedding tensor and returns its magnitude."""
        # Calculate L2 norm (magnitude) along the last dimension
        magnitude = torch.norm(embeds, p=2, dim=-1, keepdim=True)
        # Avoid division by zero for zero vectors
        normalized_embeds = embeds / (magnitude + 1e-6)
        return normalized_embeds, magnitude

    def _interpolate_embeddings(
        self,
        embeds_1: torch.Tensor | None,
        embeds_2: torch.Tensor | None,
        alpha: float,
        embedding_type: str,
    ) -> torch.Tensor | None:
        """
        Helper to interpolate embeddings, handling missing inputs and different
        interpolation methods based on `use_magnitude_separation`.
        """
        if embeds_1 is None and embeds_2 is None:
            warning(f"Both {embedding_type} embeddings are missing. Cannot interpolate.")
            return None
        elif embeds_1 is None:
            info(f"First {embedding_type} embedding is missing. Returning second embedding.")
            return embeds_2
        elif embeds_2 is None:
            info(f"Second {embedding_type} embedding is missing. Returning first embedding.")
            return embeds_1

        # Ensure tensors are on the same device
        if embeds_1.device != embeds_2.device:
            embeds_2 = embeds_2.to(embeds_1.device)

        # Align dimensions if necessary (e.g., if one is a single embedding and the other is batched)
        min_shape_len = min(embeds_1.shape[-1], embeds_2.shape[-1])
        embeds_1 = embeds_1[..., :min_shape_len]
        embeds_2 = embeds_2[..., :min_shape_len]
        
        # Ensure dimensions match for element-wise operations (if not already handled by slicing)
        if embeds_1.shape != embeds_2.shape:
             # Find the minimum shape across all dimensions except the last one for batching scenarios
            min_shape_dims = [min(d1, d2) for d1, d2 in zip(embeds_1.shape[:-1], embeds_2.shape[:-1])]
            embeds_1 = embeds_1[[slice(None, dim) for dim in min_shape_dims]]
            embeds_2 = embeds_2[[slice(None, dim) for dim in min_shape_dims]]


        normalized_embeds_1, magnitude_1 = self._normalize_and_get_magnitude(embeds_1)
        normalized_embeds_2, magnitude_2 = self._normalize_and_get_magnitude(embeds_2)

        # SLERP for direction
        interpolated_normalized = slerp(normalized_embeds_1, normalized_embeds_2, alpha)

        # LERP for magnitude
        interpolated_magnitude = (1.0 - alpha) * magnitude_1 + alpha * magnitude_2

        # Recombine
        interpolated_embeds = interpolated_normalized * interpolated_magnitude
        return interpolated_embeds.to(embeds_1.dtype)


    def invoke(self, context: InvocationContext) -> FluxConditioningBlendOutput:
        info(f"Interpolating FLUX Conditionings with alpha={self.alpha}, use_magnitude_separation={self.use_magnitude_separation}")

        conditioning_info_1 = self._load_conditioning_info(context, self.conditioning_1)
        conditioning_info_2 = self._load_conditioning_info(context, self.conditioning_2)

        if not conditioning_info_1 and not conditioning_info_2:
            error("No valid conditioning objects provided. Cannot interpolate.")
            return FluxConditioningBlendOutput(conditioning=FluxConditioningField(conditioning_name=""))

        # Extract CLIP and T5 embeddings safely
        clip_embeds_1 = conditioning_info_1.clip_embeds if conditioning_info_1 else None
        t5_embeds_1 = conditioning_info_1.t5_embeds if conditioning_info_1 else None
        clip_embeds_2 = conditioning_info_2.clip_embeds if conditioning_info_2 else None
        t5_embeds_2 = conditioning_info_2.t5_embeds if conditioning_info_2 else None

        interpolated_clip = None
        interpolated_t5 = None

        if self.use_magnitude_separation:
            info("Using magnitude separation interpolation method.")
            interpolated_clip = self._interpolate_embeddings(
                clip_embeds_1, clip_embeds_2, self.alpha, "CLIP"
            )
            interpolated_t5 = self._interpolate_embeddings(
                t5_embeds_1, t5_embeds_2, self.alpha, "T5"
            )
        else:
            info("Using direct SLERP interpolation method.")
            if clip_embeds_1 is not None and clip_embeds_2 is not None:
                # Ensure tensors are on the same device
                if clip_embeds_1.device != clip_embeds_2.device:
                    clip_embeds_2 = clip_embeds_2.to(clip_embeds_1.device)
                
                # Align dimensions for CLIP embeddings
                min_shape_clip = [min(d1, d2) for d1, d2 in zip(clip_embeds_1.shape, clip_embeds_2.shape)]
                if clip_embeds_1.dim() > 1:
                    clip_embeds_1 = clip_embeds_1[:, :min_shape_clip[1]]
                if clip_embeds_2.dim() > 1:
                    clip_embeds_2 = clip_embeds_2[:, :min_shape_clip[1]]

                interpolated_clip = slerp(clip_embeds_1, clip_embeds_2, self.alpha)
            else:
                warning("One or both CLIP embeddings are missing. Cannot interpolate CLIP embeddings directly.")
                interpolated_clip = clip_embeds_1 if clip_embeds_1 is not None else clip_embeds_2

            if t5_embeds_1 is not None and t5_embeds_2 is not None:
                # Ensure tensors are on the same device
                if t5_embeds_1.device != t5_embeds_2.device:
                    t5_embeds_2 = t5_embeds_2.to(t5_embeds_1.device)

                # Align dimensions for T5 embeddings
                min_shape_t5 = [min(d1, d2) for d1, d2 in zip(t5_embeds_1.shape, t5_embeds_2.shape)]
                if t5_embeds_1.dim() > 1:
                    t5_embeds_1 = t5_embeds_1[:, :min_shape_t5[1]]
                if t5_embeds_2.dim() > 1:
                    t5_embeds_2 = t5_embeds_2[:, :min_shape_t5[1]]

                interpolated_t5 = slerp(t5_embeds_1, t5_embeds_2, self.alpha)
            else:
                warning("One or both T5 embeddings are missing. Cannot interpolate T5 embeddings directly.")
                interpolated_t5 = t5_embeds_1 if t5_embeds_1 is not None else t5_embeds_2

        if interpolated_clip is None and interpolated_t5 is None:
            error("No embeddings were successfully interpolated. Returning an empty conditioning.")
            return FluxConditioningBlendOutput(conditioning=FluxConditioningField(conditioning_name=""))

        # Create a new FLUXConditioningInfo object with interpolated embeddings
        new_flux_conditioning_info = FLUXConditioningInfo(
            clip_embeds=interpolated_clip,
            t5_embeds=interpolated_t5,
        )

        # Wrap in ConditioningFieldData and save to context
        # The .conditionings attribute of ConditioningFieldData expects a list of FLUXConditioningInfo
        conditioning_data_to_save = ConditioningFieldData(conditionings=[new_flux_conditioning_info])
        interpolated_conditioning_name = context.conditioning.save(conditioning_data_to_save)

        info(f"Successfully interpolated FLUX Conditionings. New conditioning name: {interpolated_conditioning_name}")

        return FluxConditioningBlendOutput(
            conditioning=FluxConditioningField(conditioning_name=interpolated_conditioning_name)
        )
