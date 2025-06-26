import torch
from typing import Optional, List

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


@invocation(
    "flux_conditioning_concatenate",
    title="Concatenate Flux Conditionings",
    tags=["conditioning", "flux", "concatenate", "merge", "utility"],
    category="conditioning",
    version="1.3.0",  # Increment version on each feature change
)
class ConcatenateFluxConditioningInvocation(BaseInvocation):
    """
    Concatenates the T5 embedding tensors of up to six input Flux Conditioning objects.
    Provides flexible control over the CLIP embedding: select by index, concatenate all,
    or generate a pad tensor.
    """

    conditioning_1: FluxConditioningField | None = InputField(
        default=None,
        description="First optional Flux Conditioning input.",
     )
    strength_1: float = InputField(
        default=1.0,
        description="Strength for the first conditioning input (multiplies its embedding tensors).",
        ge=-2.0,
        le=2.0,
     )
    conditioning_2: FluxConditioningField | None = InputField(
        default=None,
        description="Second optional Flux Conditioning input.",
     )
    strength_2: float = InputField(
        default=1.0,
        description="Strength for the second conditioning input (multiplies its embedding tensors).",
        ge=-2.0,
        le=2.0,
     )
    conditioning_3: FluxConditioningField | None = InputField(
        default=None,
        description="Third optional Flux Conditioning input.",
     )
    strength_3: float = InputField(
        default=1.0,
        description="Strength for the third conditioning input (multiplies its embedding tensors).",
        ge=-2.0,
        le=2.0,
    )
    conditioning_4: FluxConditioningField | None = InputField(
        default=None,
        description="Fourth optional Flux Conditioning input.",
    )
    strength_4: float = InputField(
        default=1.0,
        description="Strength for the fourth conditioning input (multiplies its embedding tensors).",
        ge=-2.0,
        le=2.0,
    )
    conditioning_5: FluxConditioningField | None = InputField(
        default=None,
        description="Fifth optional Flux Conditioning input.",
    )
    strength_5: float = InputField(
        default=1.0,
        description="Strength for the fifth conditioning input (multiplies its embedding tensors).",
        ge=-2.0,
        le=2.0,
    )
    conditioning_6: FluxConditioningField | None = InputField(
        default=None,
        description="Sixth optional Flux Conditioning input.",
    )
    strength_6: float = InputField(
        default=1.0,
        description="Strength for the sixth conditioning input (multiplies its embedding tensors).",
        ge=-2.0,
        le=2.0,
    )
    select_clip: int = InputField(
        default=0,
        description="Index of single CLIP embedding to pass on [0-n). ",
        ge=-2,
    )

    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        """
        Loads multiple Flux Conditioning objects, concatenates their T5 embeddings,
        processes CLIP embeddings based on 'select_clip', and saves the new combined conditioning.
        """
        all_input_conditionings: List[Optional[FluxConditioningField]] = [
            self.conditioning_1,
            self.conditioning_2,
            self.conditioning_3,
            self.conditioning_4,
            self.conditioning_5,
            self.conditioning_6,
        ]

        all_input_strengths: List[float] = [
            self.strength_1,
            self.strength_2,
            self.strength_3,
            self.strength_4,
            self.strength_5,
            self.strength_6,
        ]

        loaded_flux_infos: List[FLUXConditioningInfo] = []
        loaded_flux_strengths: List[float] = [] # Store strengths corresponding to loaded infos

        # Load all provided conditionings and their corresponding strengths
        for i, cond_field in enumerate(all_input_conditionings):
            if cond_field is not None:
                info(f"Loading conditioning from input {i+1}: {cond_field.conditioning_name}")
                try:
                    conditioning_data: ConditioningFieldData = context.conditioning.load(
                        cond_field.conditioning_name
                    )
                    if not conditioning_data.conditionings:
                        warning(f"Conditioning from input {i+1} is empty or invalid. Skipping.")
                        continue
                    loaded_flux_infos.append(conditioning_data.conditionings[0])
                    loaded_flux_strengths.append(all_input_strengths[i]) # Add corresponding strength
                    info(f"Successfully loaded conditioning from input {i+1}.")
                except Exception as e:
                    error(f"Failed to load conditioning from input {i+1}: {e}")
                    raise Exception(f"Failed to load conditioning from input {i+1}: {e}")

        # Separate lists for T5 and CLIP embeddings, applying strengths
        all_t5_embeds_tensors: List[torch.Tensor] = []
        all_clip_embeds_tensors: List[torch.Tensor] = []
        
        # Populate lists and check for None, applying strength
        for i, flux_info in enumerate(loaded_flux_infos):
            strength = loaded_flux_strengths[i] # Get the strength for this specific conditioning

            if flux_info.t5_embeds is None:
                warning(f"T5 embeddings missing for conditioning from input {i+1}. Skipping for T5 concatenation.")
            else:
                # Apply strength to T5 embedding
                all_t5_embeds_tensors.append(flux_info.t5_embeds * strength)
                
            if flux_info.clip_embeds is None:
                warning(f"CLIP embeddings missing for conditioning from input {i+1}. Skipping for CLIP processing.")
            else:
                # Apply strength to CLIP embedding
                all_clip_embeds_tensors.append(flux_info.clip_embeds * strength)

        # Determine the target device from the first available CLIP or T5 embedding
        target_device = 'cpu' # Default device
        if all_clip_embeds_tensors:
            target_device = all_clip_embeds_tensors[0].device
        elif all_t5_embeds_tensors:
            target_device = all_t5_embeds_tensors[0].device


        # --- Process CLIP Embeddings based on self.select_clip ---
        final_clip_embeds: Optional[torch.Tensor] = None

        if self.select_clip == -2:
            # Return a pad tensor
            info("Returning a zero-filled (pad) CLIP tensor.")
            if all_clip_embeds_tensors:
                # Use the shape of the first valid CLIP embedding as a template
                clip_shape = all_clip_embeds_tensors[0].shape
                final_clip_embeds = torch.zeros(clip_shape, device=target_device)
            else:
                # Default shape if no CLIP embeddings are available to infer from
                # Common CLIP embedding shape: (batch_size, sequence_length, embedding_dimension)
                final_clip_embeds = torch.zeros((1, 77, 768), device=target_device) # Default to common shape
        elif self.select_clip == -1:
            # Concatenate all available CLIP embeddings
            info("Concatenating all available CLIP embeddings.")
            if len(all_clip_embeds_tensors) < 2:
                error("CLIP concatenation requires at least two valid CLIP embeddings. Only one or zero found.")
                raise Exception("CLIP concatenation requires at least two valid CLIP embeddings.")
            
            try:
                # Move all CLIP tensors to the target device before concatenation
                clip_tensors_on_device = [t.to(target_device) for t in all_clip_embeds_tensors]
                final_clip_embeds = torch.cat(clip_tensors_on_device, dim=1)
                info(f"Concatenated CLIP embeddings. Final CLIP shape: {final_clip_embeds.shape}")
            except Exception as e:
                error(f"Error concatenating CLIP embeddings: {e}")
                raise Exception(f"Error concatenating CLIP embeddings: {e}")
        elif self.select_clip >= 0:
            # Select a specific CLIP embedding by index
            info(f"Selecting CLIP embedding at index {self.select_clip}.")
            if not all_clip_embeds_tensors:
                error("No valid CLIP embeddings provided to select from.")
                raise Exception("No valid CLIP embeddings provided to select from.")
            if self.select_clip >= len(all_clip_embeds_tensors):
                error(f"CLIP selection index {self.select_clip} is out of bounds. Available CLIP embeddings: {len(all_clip_embeds_tensors)}.")
                raise Exception(f"CLIP selection index {self.select_clip} is out of bounds.")
            
            final_clip_embeds = all_clip_embeds_tensors[self.select_clip].to(target_device)
            info(f"Selected CLIP embedding at index {self.select_clip}. Final CLIP shape: {final_clip_embeds.shape}")
        
        # Ensure final_clip_embeds is not None after processing, unless it's a valid pad tensor
        if final_clip_embeds is None:
            error("Failed to determine final CLIP embeddings. This should not happen.")
            raise Exception("Failed to determine final CLIP embeddings.")


        # --- Process T5 Embeddings (Concatenate) ---
        # If all_t5_embeds_tensors is empty, the next line will raise an IndexError,
        # which is the intended behavior for no T5 inputs.
        current_t5_embeds = all_t5_embeds_tensors[0].to(target_device)

        # Concatenate subsequent T5 conditionings
        for i, t5_embeds in enumerate(all_t5_embeds_tensors[1:]): # Start from the second item
            input_number = i + 2 # Corresponds to conditioning_2, conditioning_3, etc. in original input list
            info(f"Concatenating T5 with conditioning from input {input_number}...")

            try:
                next_t5_embeds = t5_embeds.to(target_device)
                current_t5_embeds = torch.cat((current_t5_embeds, next_t5_embeds), dim=1)
                info(f"Concatenated T5 with input {input_number}. Current T5 shape: {current_t5_embeds.shape}")
            except Exception as e:
                error(f"Error concatenating T5 embeddings from input {input_number}: {e}")
                raise Exception(f"Error concatenating T5 embeddings from input {input_number}: {e}")


        # --- Create and Save New Flux Conditioning ---
        info("Creating new FLUX Conditioning object with processed CLIP and concatenated T5 embeddings...")
        new_flux_info = FLUXConditioningInfo(
            clip_embeds=final_clip_embeds,
            t5_embeds=current_t5_embeds,
        )
        new_conditioning_data = ConditioningFieldData(conditionings=[new_flux_info])

        try:
            new_conditioning_name = context.conditioning.save(new_conditioning_data)
            info(f"New concatenated conditioning saved with name: {new_conditioning_name}")
        except Exception as e:
            error(f"Failed to save new concatenated conditioning: {e}")
            raise Exception(f"Failed to save new concatenated conditioning: {e}")

        # --- Return the new Flux Conditioning output ---
        return FluxConditioningOutput.build(conditioning_name=new_conditioning_name)
