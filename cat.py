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
    # Removed InvocationError as per user feedback
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
    version="1.1.1",  # Incrementing version due to CLIP concatenation logic change
)
class ConcatenateFluxConditioningInvocation(BaseInvocation):
    """
    Concatenates the T5 embedding tensors of up to six input Flux Conditioning objects.
    The CLIP embedding from the first provided conditioning is used as the final CLIP embedding.
    Outputs a new Flux Conditioning object with the combined embeddings.
    Requires at least two conditioning inputs to perform concatenation.
    """

    conditioning_1: FluxConditioningField | None = InputField(
        default=None,
        description="First optional Flux Conditioning input. Its CLIP embedding will be used.",
        ui_order=0,
    )
    conditioning_2: FluxConditioningField | None = InputField(
        default=None,
        description="Second optional Flux Conditioning input. Its T5 embedding will be concatenated.",
        ui_order=1,
    )
    conditioning_3: FluxConditioningField | None = InputField(
        default=None,
        description="Third optional Flux Conditioning input. Its T5 embedding will be concatenated.",
        ui_order=2,
    )
    conditioning_4: FluxConditioningField | None = InputField(
        default=None,
        description="Fourth optional Flux Conditioning input. Its T5 embedding will be concatenated.",
        ui_order=3,
    )
    conditioning_5: FluxConditioningField | None = InputField(
        default=None,
        description="Fifth optional Flux Conditioning input. Its T5 embedding will be concatenated.",
        ui_order=4,
    )
    conditioning_6: FluxConditioningField | None = InputField(
        default=None,
        description="Sixth optional Flux Conditioning input. Its T5 embedding will be concatenated.",
        ui_order=5,
    )

    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        """
        Loads multiple Flux Conditioning objects, concatenates their T5 embeddings,
        uses the first CLIP embedding, and saves the new combined conditioning.
        """
        all_input_conditionings: List[Optional[FluxConditioningField]] = [
            self.conditioning_1,
            self.conditioning_2,
            self.conditioning_3,
            self.conditioning_4,
            self.conditioning_5,
            self.conditioning_6,
        ]

        loaded_flux_infos: List[FLUXConditioningInfo] = []

        # Load all provided conditionings
        for i, cond_field in enumerate(all_input_conditionings):
            if cond_field is not None:
                info(f"Loading conditioning from input {i+1}: {cond_field.conditioning_name}")
                try:
                    conditioning_data: ConditioningFieldData = context.conditioning.load(
                        cond_field.conditioning_name
                    )
                    if not conditioning_data.conditionings:
                        # Continue to next if the loaded data itself is empty, don't break loop
                        warning(f"Conditioning from input {i+1} is empty or invalid. Skipping.")
                        continue
                    loaded_flux_infos.append(conditioning_data.conditionings[0])
                    info(f"Successfully loaded conditioning from input {i+1}.")
                except Exception as e:
                    error(f"Failed to load conditioning from input {i+1}: {e}")
                    raise Exception(f"Failed to load conditioning from input {i+1}: {e}")

        if len(loaded_flux_infos) < 2:
            error(
                "Concatenation requires at least two valid Flux Conditioning inputs. "
                f"Only {len(loaded_flux_infos)} were provided."
            )
            raise Exception(
                "Concatenation requires at least two valid Flux Conditioning inputs."
            )

        # Initialize with the first conditioning's embeddings
        # For CLIP, we now just take the first one
        final_clip_embeds = loaded_flux_infos[0].clip_embeds
        current_t5_embeds = loaded_flux_infos[0].t5_embeds

        if final_clip_embeds is None:
            error("CLIP embeddings missing from the first valid conditioning. Cannot proceed.")
            raise Exception("CLIP embeddings missing from the first valid conditioning.")
        if current_t5_embeds is None:
            error("T5 embeddings missing from the first valid conditioning. Cannot proceed with concatenation.")
            raise Exception("T5 embeddings missing from the first valid conditioning.")

        device = final_clip_embeds.device # Use the device from the first CLIP embed

        # Concatenate subsequent T5 conditionings only
        for i, flux_info in enumerate(loaded_flux_infos[1:]): # Start from the second item
            input_number = i + 2 # Corresponds to conditioning_2, conditioning_3, etc.
            info(f"Concatenating T5 with conditioning from input {input_number}...")

            if flux_info.t5_embeds is None:
                warning(f"T5 embeddings missing for conditioning from input {input_number}. Skipping T5 concatenation for this input.")
                continue # Skip this specific input's T5 if it's missing

            try:
                # Move to the same device if not already
                next_t5_embeds = flux_info.t5_embeds.to(device)
                current_t5_embeds = torch.cat((current_t5_embeds, next_t5_embeds), dim=1)
                info(f"Concatenated T5 with input {input_number}. Current T5 shape: {current_t5_embeds.shape}")
            except Exception as e:
                error(f"Error concatenating T5 embeddings from input {input_number}: {e}")
                raise Exception(f"Error concatenating T5 embeddings from input {input_number}: {e}")

        # --- Create and Save New Flux Conditioning ---
        info("Creating new FLUX Conditioning object with first CLIP and concatenated T5 embeddings...")
        new_flux_info = FLUXConditioningInfo(
            clip_embeds=final_clip_embeds, # Use the CLIP from the first input
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

