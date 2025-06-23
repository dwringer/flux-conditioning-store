import sqlite3
import io
import os

import torch

from typing import Union, Optional

from invokeai.invocation_api import (
    BaseInvocation,
    InputField,
    InvocationContext,
    invocation,
    OutputField,
    invocation_output,
    BaseInvocationOutput,
)
from invokeai.app.invocations.primitives import (
    FluxConditioningOutput,
)
from invokeai.app.invocations.fields import(
    FluxConditioningField,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    FLUXConditioningInfo,
    ConditioningFieldData,
)
from invokeai.backend.util.logging import info, warning, error


# Define the database file name (must be the same as the store node)
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flux_conditionings.db")

# Define a custom output class that includes both a single conditioning and a list
@invocation_output("retrieve_flux_conditioning_multi_output")
class RetrieveFluxConditioningMultiOutput(BaseInvocationOutput):
    """
    Output for the Retrieve Flux Conditioning node, providing a single selected
    conditioning and a list of all retrieved conditionings.
    """
    conditioning: FluxConditioningField = OutputField(
        description="A single selected Flux Conditioning (selected by index from the retrieved list)"
    )
    conditioning_list: list[FluxConditioningField] = OutputField(
        description="A list of all retrieved Flux Conditionings"
    )

@invocation(
    "retrieve_flux_conditioning",
    title="Retrieve Flux Conditioning",
    tags=["conditioning", "flux", "database", "retrieve", "list"],
    category="conditioning",
    version="1.1.2", # Incrementing version due to feature change
)
class RetrieveFluxConditioningInvocation(BaseInvocation):
    """
    Retrieves one or more FLUX Conditioning objects (CLIP and T5 embeddings)
    from an SQLite database using unique identifiers.
    Outputs a single selected conditioning and a list of all retrieved conditionings.
    """

    conditioning_id_or_list: Union[str, list[str]] = InputField(
        description="The unique identifier(s) of the Flux Conditioning(s) to retrieve.",
        ui_order=0,
    )
    
    select_index: int = InputField(
        default=0,
        description="Index of the retrieved conditioning to output as the single 'conditioning' field. If out of bounds, uses modulus.",
        ui_order=1,
    )

    def invoke(self, context: InvocationContext) -> RetrieveFluxConditioningMultiOutput:
        """
        Main invocation method to retrieve the conditioning(s).
        """
        # Ensure conditioning_ids is always a list for consistent processing
        conditioning_ids_to_retrieve: list[str] = []
        if isinstance(self.conditioning_id_or_list, str):
            conditioning_ids_to_retrieve = [self.conditioning_id_or_list]
        elif isinstance(self.conditioning_id_or_list, list):
            conditioning_ids_to_retrieve = self.conditioning_id_or_list
        else:
            error("Invalid input type for conditioning_id_or_list. Must be a string or a list of strings.")
            # Return empty outputs if input is invalid
            return RetrieveFluxConditioningMultiOutput(
                conditioning=FluxConditioningField(conditioning_name=""),
                conditioning_list=[]
            )

        if not conditioning_ids_to_retrieve:
            warning("No conditioning IDs provided for retrieval. Returning empty outputs.")
            return RetrieveFluxConditioningMultiOutput(
                conditioning=FluxConditioningField(conditioning_name=""),
                conditioning_list=[]
            )

        retrieved_flux_conditioning_fields: list[FluxConditioningField] = []
        
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()

            # Construct the query for multiple IDs using the IN clause
            placeholders = ",".join("?" * len(conditioning_ids_to_retrieve))
            query = f"SELECT id, clip_embeds, t5_embeds FROM flux_conditionings WHERE id IN ({placeholders})"
            
            cursor.execute(query, conditioning_ids_to_retrieve)
            results = cursor.fetchall() # Fetch all results in one go
            conn.close() # Close connection after fetching results

            retrieved_ids_set = {row[0] for row in results}
            missing_ids = [cond_id for cond_id in conditioning_ids_to_retrieve if cond_id not in retrieved_ids_set]
            for missing_id in missing_ids:
                warning(f"No conditioning found with ID: {missing_id}")


            for result_row in results:
                cond_id, clip_bytes, t5_bytes = result_row

                clip_embeds_tensor = None
                t5_embeds_tensor = None

                # Deserialize tensors from bytes
                try:
                    clip_buffer = io.BytesIO(clip_bytes)
                    clip_embeds_tensor = torch.load(clip_buffer)

                    t5_buffer = io.BytesIO(t5_bytes)
                    t5_embeds_tensor = torch.load(t5_buffer)
                except Exception as e:
                    error(f"Failed to deserialize tensors for ID {cond_id}: {e}")
                    # Continue to next ID if deserialization fails for one
                    continue 

                if clip_embeds_tensor is None or t5_embeds_tensor is None:
                    error(f"Retrieved conditioning for ID {cond_id} is incomplete. Skipping.")
                    # Continue to next ID if incomplete
                    continue

                # Reconstruct and save the FLUX Conditioning object to the InvokeAI context
                conditioning_info = FLUXConditioningInfo(clip_embeds=clip_embeds_tensor, t5_embeds=t5_embeds_tensor)
                conditioning_data = ConditioningFieldData(conditionings=[conditioning_info])
                
                # This will save the conditioning to InvokeAI's internal memory management system
                # and return a name (ID) that can be used by other nodes.
                conditioning_name = context.conditioning.save(conditioning_data)
                info(f"Retrieved and re-saved conditioning with ID: {cond_id} as new conditioning name: {conditioning_name}")

                retrieved_flux_conditioning_fields.append(
                    FluxConditioningField(conditioning_name=conditioning_name)
                )

        except sqlite3.Error as e:
            error(f"Error accessing database during retrieval: {e}")
            return RetrieveFluxConditioningMultiOutput(
                conditioning=FluxConditioningField(conditioning_name=""),
                conditioning_list=[]
            )

        # Determine the single output conditioning using select_index
        selected_conditioning_output: FluxConditioningField
        if retrieved_flux_conditioning_fields:
            # Use modulus to ensure the index is always within bounds
            effective_index = self.select_index % len(retrieved_flux_conditioning_fields)
            selected_conditioning_output = retrieved_flux_conditioning_fields[effective_index]
            info(f"Selected conditioning at index {self.select_index} (effective index {effective_index}) for single output.")
        else:
            warning("No conditionings were successfully retrieved. Returning empty outputs for both fields.")
            selected_conditioning_output = FluxConditioningField(conditioning_name="")
            
        return RetrieveFluxConditioningMultiOutput(
            conditioning=selected_conditioning_output,
            conditioning_list=retrieved_flux_conditioning_fields
        )

