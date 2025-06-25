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
from invokeai.backend.util.logging import info, warning, error


@invocation_output("flux_conditioning_list_output")
class FluxConditioningListOutput(BaseInvocationOutput):
    """
    Output for the Flux Conditioning List node, providing an ordered list
    of Flux Conditioning objects.
    """
    conditioning_list: list[FluxConditioningField] = OutputField(
        description="An ordered list of provided Flux Conditioning objects."
    )


@invocation(
    "flux_conditioning_list",
    title="Flux Conditioning List",
    tags=["conditioning", "flux", "list", "utility", "order"],
    category="conditioning",
    version="1.0.1",
)
class FluxConditioningListInvocation(BaseInvocation):
    """
    Takes multiple optional Flux Conditioning inputs and outputs them as a single
    ordered list. Missing (None) inputs are gracefully handled.
    """

    # Define up to 6 optional Flux Conditioning inputs
    conditioning_1: FluxConditioningField | None = InputField(
        description="First optional Flux Conditioning input.",
        default=None,
        ui_order=0,
    )
    conditioning_2: FluxConditioningField | None = InputField(
        description="Second optional Flux Conditioning input.",
        default=None,
        ui_order=1,
    )
    conditioning_3: FluxConditioningField | None = InputField(
        description="Third optional Flux Conditioning input.",
        default=None,
        ui_order=2,
    )
    conditioning_4: FluxConditioningField | None = InputField(
        description="Fourth optional Flux Conditioning input.",
        default=None,
        ui_order=3,
    )
    conditioning_5: FluxConditioningField | None = InputField(
        description="Fifth optional Flux Conditioning input.",
        default=None,
        ui_order=4,
    )
    conditioning_6: FluxConditioningField | None = InputField(
        description="Sixth optional Flux Conditioning input.",
        default=None,
        ui_order=5,
    )

    def invoke(self, context: InvocationContext) -> FluxConditioningListOutput:
        """
        Collects all provided Flux Conditioning inputs into an ordered list.
        """
        # Create a list to hold all non-None conditioning inputs
        ordered_conditionings: list[FluxConditioningField] = []

        # Iterate through the inputs in their defined ui_order
        # We collect them in a list based on their field names directly,
        # which respects the ui_order implicitly defined by their numbering.
        input_fields = [
            self.conditioning_1,
            self.conditioning_2,
            self.conditioning_3,
            self.conditioning_4,
            self.conditioning_5,
            self.conditioning_6,
        ]

        for i, cond in enumerate(input_fields):
            if cond is not None:
                ordered_conditionings.append(cond)
                info(f"Added conditioning from input {i+1} to the list: {cond.conditioning_name}")
            else:
                info(f"Input conditioning {i+1} was not provided (None). Skipping.")

        if not ordered_conditionings:
            warning("No Flux Conditioning inputs were provided. The output list will be empty.")
            # Return an empty list if no conditionings are present
            return FluxConditioningListOutput(conditioning_list=[])

        info(f"Successfully compiled {len(ordered_conditionings)} Flux Conditionings into an ordered list.")
        return FluxConditioningListOutput(conditioning_list=ordered_conditionings)

