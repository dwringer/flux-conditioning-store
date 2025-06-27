import torch
from typing import Optional, List, Tuple

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
    version="1.4.1",  # Increment version on each feature change
)
class ConcatenateFluxConditioningInvocation(BaseInvocation):
    """
    Concatenates the T5 embedding tensors of up to six input Flux Conditioning objects.
    Provides flexible control over the CLIP embedding: select by 1-indexed input number,
    or generate a zeros tensor.
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
        default=1,
        description="CLIP embedding selection: 0 for a zeros tensor; 1-6 to select a specific input (1-indexed). If a selected input is missing, it falls back to the next subsequent, then preceding, available CLIP embedding.",
        ge=-1, # Changed from -2 to -1 to reflect new behavior (0 is now for zeros tensor)
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

        # Store (original_input_num, flux_info, strength) for connected inputs
        connected_inputs: List[Tuple[int, FLUXConditioningInfo, float]] = []

        # Load all provided conditionings and their corresponding strengths, keeping track of original input index
        for i, cond_field in enumerate(all_input_conditionings):
            original_input_num = i + 1 # 1-indexed input number
            if cond_field is not None:
                info(f"Loading conditioning from input {original_input_num}: {cond_field.conditioning_name}")
                try:
                    conditioning_data: ConditioningFieldData = context.conditioning.load(
                        cond_field.conditioning_name
                    )
                    if not conditioning_data.conditionings:
                        warning(f"Conditioning from input {original_input_num} is empty or invalid. Skipping.")
                        continue
                    connected_inputs.append(
                        (original_input_num, conditioning_data.conditionings[0], all_input_strengths[i])
                    )
                    info(f"Successfully loaded conditioning from input {original_input_num}.")
                except Exception as e:
                    error(f"Failed to load conditioning from input {original_input_num}: {e}")
                    raise Exception(f"Failed to load conditioning from input {original_input_num}: {e}")

        # Separate into T5 and CLIP, applying strength.
        # `clip_tensors_with_original_indices` is used for 1-indexed selection fallback.
        # `all_clip_embeds_tensors_for_concat` is a simple list of tensors for concatenation.
        all_t5_embeds_tensors: List[torch.Tensor] = []
        clip_tensors_with_original_indices: List[Tuple[int, torch.Tensor]] = []
        all_clip_embeds_tensors_for_concat: List[torch.Tensor] = []

        for original_input_num, flux_info, strength in connected_inputs:
            if flux_info.t5_embeds is None:
                warning(f"T5 embeddings missing for conditioning from input {original_input_num}. Skipping for T5 concatenation.")
            else:
                all_t5_embeds_tensors.append(flux_info.t5_embeds * strength)

            if flux_info.clip_embeds is None:
                warning(f"CLIP embeddings missing for conditioning from input {original_input_num}. Skipping for CLIP processing.")
            else:
                applied_clip_tensor = flux_info.clip_embeds * strength
                clip_tensors_with_original_indices.append((original_input_num, applied_clip_tensor))
                all_clip_embeds_tensors_for_concat.append(applied_clip_tensor)

        # Determine the target device from the first available CLIP or T5 embedding
        target_device = 'cpu' # Default device
        if all_clip_embeds_tensors_for_concat:
            target_device = all_clip_embeds_tensors_for_concat[0].device
        elif all_t5_embeds_tensors:
            target_device = all_t5_embeds_tensors[0].device


        # --- Process CLIP Embeddings based on self.select_clip ---
        final_clip_embeds: Optional[torch.Tensor] = None

        if self.select_clip == 0:
            # Output a zeros tensor
            info("Returning a zero-filled (pad) CLIP tensor as select_clip is 0.")
            if all_clip_embeds_tensors_for_concat:
                # Use the shape of the first valid CLIP embedding as a template if available
                clip_shape = all_clip_embeds_tensors_for_concat[0].shape
                final_clip_embeds = torch.zeros(clip_shape, device=target_device)
            else:
                # Default to common CLIP embedding shape if no inputs are connected to infer from
                # Common CLIP embedding shape: (batch_size, sequence_length, embedding_dimension)
                final_clip_embeds = torch.zeros((1, 77, 768), device=target_device)
        elif self.select_clip == -1:
            # Concatenate all available CLIP embeddings
            info("Concatenating all available CLIP embeddings.")
            if not all_clip_embeds_tensors_for_concat:
                error("No valid CLIP embeddings provided for concatenation.")
                raise Exception("No valid CLIP embeddings provided for concatenation.")
           
            if len(all_clip_embeds_tensors_for_concat) == 1:
                info("Only one CLIP embedding found. Returning it directly as concatenation of a single tensor.")
                final_clip_embeds = all_clip_embeds_tensors_for_concat[0].to(target_device)
            else:
                try:
                    # Move all CLIP tensors to the target device before concatenation
                    clip_tensors_on_device = [t.to(target_device) for t in all_clip_embeds_tensors_for_concat]
                    final_clip_embeds = torch.cat(clip_tensors_on_device, dim=1)
                    info(f"Concatenated CLIP embeddings. Final CLIP shape: {final_clip_embeds.shape}")
                except Exception as e:
                    error(f"Error concatenating CLIP embeddings: {e}")
                    raise Exception(f"Error concatenating CLIP embeddings: {e}")
        elif self.select_clip > 0:
            # Select a specific CLIP embedding by 1-indexed input number, with fallback logic
            requested_input_num = self.select_clip
            info(f"Attempting to select CLIP embedding from input {requested_input_num} (1-indexed).")

            if not clip_tensors_with_original_indices:
                error("No valid CLIP embeddings provided from any input to select from.")
                raise Exception("No valid CLIP embeddings provided from any input to select from.")

            selected_clip_tensor = None

            # 1. Try to find the exact requested input first
            for original_num, clip_tensor in clip_tensors_with_original_indices:
                if original_num == requested_input_num:
                    selected_clip_tensor = clip_tensor
                    info(f"Found CLIP embedding directly from requested input {requested_input_num}.")
                    break

            # 2. If not found, search forward for the next subsequent available CLIP embedding
            if selected_clip_tensor is None:
                info(f"CLIP embedding not found directly for input {requested_input_num}. Searching for next subsequent available.")
                for original_num, clip_tensor in clip_tensors_with_original_indices:
                    if original_num > requested_input_num:
                        selected_clip_tensor = clip_tensor
                        info(f"Found CLIP embedding from next subsequent input {original_num}.")
                        break
           
            # 3. If still not found, search backward for the preceding available CLIP embedding
            if selected_clip_tensor is None:
                info(f"No subsequent CLIP embedding found. Searching for preceding available.")
                # Iterate in reverse order to find the largest original_num that is less than requested_input_num,
                # effectively finding the closest preceding input.
                for original_num, clip_tensor in reversed(clip_tensors_with_original_indices):
                    if original_num < requested_input_num:
                        selected_clip_tensor = clip_tensor
                        info(f"Found CLIP embedding from preceding input {original_num}.")
                        break

            if selected_clip_tensor is None:
                # This case indicates that despite having available CLIP embeddings,
                # none could be found matching the selection criteria (exact, subsequent, or preceding).
                # This scenario should be rare if 'clip_tensors_with_original_indices' is not empty.
                error(f"Could not find a suitable CLIP embedding for selected input {requested_input_num} despite available inputs. This indicates an unexpected state in selection logic.")
                raise Exception(f"Could not find a suitable CLIP embedding for selected input {requested_input_num}.")
           
            final_clip_embeds = selected_clip_tensor.to(target_device)
            info(f"Selected CLIP embedding. Final CLIP shape: {final_clip_embeds.shape}")
        else:
            # Catch any other invalid select_clip values (e.g., negative values other than -1)
            error(f"Invalid select_clip value: {self.select_clip}. Must be 0 (zeros) or a positive integer (1-indexed input selection).")
            raise Exception(f"Invalid select_clip value: {self.select_clip}.")

        # Ensure final_clip_embeds is not None after processing
        if final_clip_embeds is None:
            error("Failed to determine final CLIP embeddings. This should not happen after the selection logic has been applied.")
            raise Exception("Failed to determine final CLIP embeddings.")


        # --- Process T5 Embeddings (Concatenate) ---
        # If all_t5_embeds_tensors is empty, the next line will raise an IndexError,
        # which means no valid T5 inputs were provided, and is an intended behavior
        # as there's nothing to concatenate.
        if not all_t5_embeds_tensors:
            error("No valid T5 embeddings found to concatenate. At least one T5 input is required.")
            raise Exception("No valid T5 embeddings found to concatenate.")
           
        current_t5_embeds = all_t5_embeds_tensors[0].to(target_device)

        # Concatenate subsequent T5 conditionings
        for i, t5_embeds in enumerate(all_t5_embeds_tensors[1:]): # Start from the second item if more exist
            # The original_input_num for these starts from the input that provided this t5_embed.
            # We don't explicitly need it here since it's just concatenation.
            info(f"Concatenating T5 embeddings...")

            try:
                next_t5_embeds = t5_embeds.to(target_device)
                current_t5_embeds = torch.cat((current_t5_embeds, next_t5_embeds), dim=1)
                info(f"Concatenated T5. Current T5 shape: {current_t5_embeds.shape}")
            except Exception as e:
                error(f"Error concatenating T5 embeddings: {e}")
                raise Exception(f"Error concatenating T5 embeddings: {e}")


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
    
