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
from invokeai.app.invocations.fields import (
    FluxConditioningField,
)
from invokeai.app.invocations.primitives import (
    FluxConditioningOutput,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    FLUXConditioningInfo,
    ConditioningFieldData,
)
from invokeai.backend.util.logging import info, warning, error

# Define a custom output class for the Conditioning Delta and Augmented Conditioning
@invocation_output("conditioning_delta_and_augmented_output")
class FluxConditioningDeltaAndAugmentedOutput(BaseInvocationOutput):
    """Output for the Conditioning Delta and Augmented Conditioning."""
    augmented_conditioning: FluxConditioningField = OutputField(description="The augmented conditioning (base + delta, or just delta)", ui_order=1)
    delta_conditioning: FluxConditioningField = OutputField(description="The resulting conditioning delta (feature - reference)", ui_order=2)

def _average_conditioning_list(
    conditioning_list: list[FLUXConditioningInfo]
) -> FLUXConditioningInfo:
    """Averages a list of FLUXConditioningInfo objects into a single one."""
    if not conditioning_list:
        raise ValueError("Cannot average an empty list of conditionings.")

    # Stack CLIP embeddings
    clip_embeds = [c.clip_embeds for c in conditioning_list if c.clip_embeds is not None]
    if not clip_embeds:
        raise ValueError("No CLIP embeddings found in the conditioning list to average.")
    avg_clip_embeds = torch.stack(clip_embeds).mean(dim=0)

    # Stack T5 embeddings if they exist in any conditioning
    t5_embeds = [c.t5_embeds for c in conditioning_list if c.t5_embeds is not None]
    avg_t5_embeds = None
    if t5_embeds:
        avg_t5_embeds = torch.stack(t5_embeds).mean(dim=0)

    return FLUXConditioningInfo(clip_embeds=avg_clip_embeds, t5_embeds=avg_t5_embeds)

@invocation(
    "flux_conditioning_delta_augmentation",
    title="Flux Conditioning Delta",
    tags=["conditioning", "flux", "arithmetic", "delta", "augment"],
    category="conditioning",
    version="1.0.0",
)
class FluxConditioningDeltaAugmentationInvocation(BaseInvocation):
    """
    Calculates the delta between feature and reference conditionings,
    and optionally augments a base conditioning with this delta.
    If reference conditioning is omitted, it will be treated as zero tensors.
    """

    feature_conditioning: FluxConditioningField | list[FluxConditioningField] = InputField(
        description="Feature Conditioning (single or list) for delta calculation. If a list, it will be averaged.",
        ui_order=1,
    )
    reference_conditioning: FluxConditioningField | list[FluxConditioningField] | None = InputField(
        default=None,
        description="Reference Conditioning (single or list) for delta calculation. If a list, it will be averaged. If omitted, zero tensors will be used as reference.",
        ui_order=2,
    )
    base_conditioning: FluxConditioningField | None = InputField(
        default=None,
        description="Optional Base Conditioning to which the delta will be added. If not provided, Augmented Conditioning will be the Delta.",
        ui_order=3,
    )
    base_scale: float = InputField(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Scalar to multiply the base conditioning when augmenting.",
        ui_order=4,
    )
    delta_scale: float = InputField(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Scalar to multiply the delta when augmenting the base conditioning.",
        ui_order=5,
    )
    scale_delta_output: bool = InputField(
        default=False,
        description="If true, the delta output will also be scaled by the delta_scale.",
        ui_order=6,
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

    def invoke(self, context: InvocationContext) -> FluxConditioningDeltaAndAugmentedOutput:
        # Resolve feature conditioning (always required)
        feature_avg_info: FLUXConditioningInfo
        if isinstance(self.feature_conditioning, list):
            feature_cond_infos = []
            for fc_field in self.feature_conditioning:
                loaded_info = self._load_conditioning_info(context, fc_field)
                if loaded_info: # Ensure loaded_info is not None
                    feature_cond_infos.append(loaded_info)
            if not feature_cond_infos:
                raise ValueError("No valid feature conditionings could be loaded from the provided list.")
            feature_avg_info = _average_conditioning_list(feature_cond_infos)
            info(f"Averaged {len(feature_cond_infos)} feature conditionings.")
        else:
            feature_avg_info = self._load_conditioning_info(context, self.feature_conditioning)
            if feature_avg_info is None:
                raise ValueError("Failed to load single feature conditioning.")
            info("Using single feature conditioning.")

        # Resolve reference conditioning (can be None)
        reference_avg_info: FLUXConditioningInfo
        if self.reference_conditioning is None:
            info("No reference conditioning provided. Substituting zero tensors for delta calculation.")
            # Create zero tensors matching the shape of feature_avg_info's embeddings
            zero_clip_embeds = torch.zeros_like(feature_avg_info.clip_embeds)
            zero_t5_embeds = None
            if feature_avg_info.t5_embeds is not None:
                zero_t5_embeds = torch.zeros_like(feature_avg_info.t5_embeds)
            reference_avg_info = FLUXConditioningInfo(
                clip_embeds=zero_clip_embeds,
                t5_embeds=zero_t5_embeds,
            )
        elif isinstance(self.reference_conditioning, list):
            reference_cond_infos = []
            for rc_field in self.reference_conditioning:
                loaded_info = self._load_conditioning_info(context, rc_field)
                if loaded_info: # Ensure loaded_info is not None
                    reference_cond_infos.append(loaded_info)
            if not reference_cond_infos:
                # If a list was provided but no valid conditionings loaded, treat as if None was provided
                info("Provided reference conditioning list was empty or contained no valid conditionings. Substituting zero tensors.")
                zero_clip_embeds = torch.zeros_like(feature_avg_info.clip_embeds)
                zero_t5_embeds = None
                if feature_avg_info.t5_embeds is not None:
                    zero_t5_embeds = torch.zeros_like(feature_avg_info.t5_embeds)
                reference_avg_info = FLUXConditioningInfo(
                    clip_embeds=zero_clip_embeds,
                    t5_embeds=zero_t5_embeds,
                )
            else:
                reference_avg_info = _average_conditioning_list(reference_cond_infos)
                info(f"Averaged {len(reference_cond_infos)} reference conditionings.")
        else: # Single FluxConditioningField provided
            reference_avg_info = self._load_conditioning_info(context, self.reference_conditioning)
            if reference_avg_info is None:
                # If a single field was provided but failed to load, treat as if None was provided
                info("Provided single reference conditioning failed to load. Substituting zero tensors.")
                zero_clip_embeds = torch.zeros_like(feature_avg_info.clip_embeds)
                zero_t5_embeds = None
                if feature_avg_info.t5_embeds is not None:
                    zero_t5_embeds = torch.zeros_like(feature_avg_info.t5_embeds)
                reference_avg_info = FLUXConditioningInfo(
                    clip_embeds=zero_clip_embeds,
                    t5_embeds=zero_t5_embeds,
                )
            else:
                info("Using single reference conditioning.")

        # --- Calculate Delta Conditioning ---
        # Ensure that both clip_embeds are not None before subtraction
        if feature_avg_info.clip_embeds is None or reference_avg_info.clip_embeds is None:
            raise ValueError("CLIP embeddings are missing for delta calculation.")
            
        delta_clip_embeds = feature_avg_info.clip_embeds - reference_avg_info.clip_embeds

        delta_t5_embeds = None
        if feature_avg_info.t5_embeds is not None and reference_avg_info.t5_embeds is not None:
            delta_t5_embeds = feature_avg_info.t5_embeds - reference_avg_info.t5_embeds
            info("T5 embeddings found and subtracted for delta.")
        elif feature_avg_info.t5_embeds is not None or reference_avg_info.t5_embeds is not None:
            warning("One of the conditionings has T5 embeds but the other doesn't. T5 delta will not be computed.")
        else:
            info("No T5 embeddings found in either conditioning for delta.")

        delta_conditioning_info = FLUXConditioningInfo(
            clip_embeds=delta_clip_embeds,
            t5_embeds=delta_t5_embeds,
        )

        # Apply delta_scale to delta_conditioning_info for augmentation/direct output
        scaled_delta_clip_embeds = delta_conditioning_info.clip_embeds * self.delta_scale
        scaled_delta_t5_embeds = None
        if delta_conditioning_info.t5_embeds is not None:
            scaled_delta_t5_embeds = delta_conditioning_info.t5_embeds * self.delta_scale

        scaled_delta_conditioning_for_augment = FLUXConditioningInfo(
            clip_embeds=scaled_delta_clip_embeds,
            t5_embeds=scaled_delta_t5_embeds,
        )        

        # --- Calculate Augmented Conditioning ---
        augmented_conditioning_info: FLUXConditioningInfo

        if self.base_conditioning:
            base_conditioning_info = self._load_conditioning_info(context, self.base_conditioning)
            if base_conditioning_info is None:
                warning("Base conditioning provided but failed to load. Augmented conditioning will be the scaled delta.")
                augmented_conditioning_info = scaled_delta_conditioning_for_augment
            else:
                info("Base conditioning provided. Calculating augmented conditioning (base + delta).")

                # Apply base_scale to base_conditioning_info
                scaled_base_clip_embeds = base_conditioning_info.clip_embeds * self.base_scale
                scaled_base_t5_embeds = None
                if base_conditioning_info.t5_embeds is not None:
                    scaled_base_t5_embeds = base_conditioning_info.t5_embeds * self.base_scale
                
                scaled_base_conditioning_info = FLUXConditioningInfo(
                    clip_embeds=scaled_base_clip_embeds,
                    t5_embeds=scaled_base_t5_embeds,
                )

                augmented_clip_embeds = scaled_base_conditioning_info.clip_embeds + scaled_delta_conditioning_for_augment.clip_embeds

                augmented_t5_embeds = None
                if scaled_base_conditioning_info.t5_embeds is not None and scaled_delta_conditioning_for_augment.t5_embeds is not None:
                    augmented_t5_embeds = scaled_base_conditioning_info.t5_embeds + scaled_delta_conditioning_for_augment.t5_embeds
                    info("T5 embeddings added for augmented conditioning.")
                elif scaled_base_conditioning_info.t5_embeds is not None:
                    augmented_t5_embeds = scaled_base_conditioning_info.t5_embeds # Use scaled base T5 if scaled_delta has none
                    warning("Scaled base conditioning has T5 embeds, but scaled delta does not. Using scaled base T5 for augmented.")
                elif scaled_delta_conditioning_for_augment.t5_embeds is not None:
                    augmented_t5_embeds = scaled_delta_conditioning_for_augment.t5_embeds # Use scaled_delta T5 if base has none
                    warning("Scaled delta has T5 embeds, but scaled base conditioning does not. Using scaled delta T5 for augmented.")
                else:
                    info("No T5 embeddings to augment.")

                augmented_conditioning_info = FLUXConditioningInfo(
                    clip_embeds=augmented_clip_embeds,
                    t5_embeds=augmented_t5_embeds,
                )
        else:
            info("No base conditioning provided. Augmented conditioning will be the scaled delta conditioning.")
            augmented_conditioning_info = scaled_delta_conditioning_for_augment

        # Save the new conditionings
        final_delta_to_save_info = delta_conditioning_info # Default to unscaled delta

        if self.scale_delta_output:
            final_delta_to_save_info = scaled_delta_conditioning_for_augment # If scaling delta output, use the scaled version
            info(f"Delta output is being scaled by {self.delta_scale}.")
        else:
            info("Delta output is not scaled.")

        delta_conditioning_data = ConditioningFieldData(conditionings=[final_delta_to_save_info])
        delta_conditioning_name = context.conditioning.save(delta_conditioning_data)
        info(f"Generated conditioning delta: {delta_conditioning_name}")

        augmented_conditioning_data = ConditioningFieldData(conditionings=[augmented_conditioning_info])
        augmented_conditioning_name = context.conditioning.save(augmented_conditioning_data)
        info(f"Generated augmented conditioning: {augmented_conditioning_name}")

        return FluxConditioningDeltaAndAugmentedOutput(
            augmented_conditioning=FluxConditioningField(conditioning_name=augmented_conditioning_name),
            delta_conditioning=FluxConditioningField(conditioning_name=delta_conditioning_name),
        )
