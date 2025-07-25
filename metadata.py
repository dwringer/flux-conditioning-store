from typing import Any

from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    ImageField,
    InputField,
    InvocationContext,
    OutputField,
    StringCollectionOutput,
    invocation,
    invocation_output,
)
from invokeai.backend.util.logging import warning, error


@invocation(
    "extract_image_collection_metadata_item",  # Unique, lowercase, underscore-separated invocation name
    title="Extract Image Collection Metadata Item",  # User-friendly title for the UI
    tags=["image", "metadata", "extraction", "collection", "utility"],  # Searchable keywords
    category="metadata",  # Category for UI organization
    version="1.0.0",  # Increment this version when making changes, especially to inputs/outputs
)
class ExtractImageCollectionMetadataItemInvocation(BaseInvocation):
    """
    This node extracts specified metadata values from a collection of input images.
    It takes an image collection and a metadata key string input.
    For each image in the collection, it attempts to retrieve the value associated
    with the provided key. The extracted values are then compiled into a string 
    collection. If a key is not found for a particular image, an empty string is 
    appended to maintain collection length.
    """

    # Input Field 1: Image Collection
    # This input accepts a list of ImageField objects, representing an image collection.
    images: list[ImageField] = InputField(
        description="A collection of images from which to extract metadata.",
        title="Image Collection",
        ui_order=0,  # Control display order in the UI
    )

    # Input Field 2: Metadata Key
    # This string input allows the user to specify which metadata key to extract.
    # It has a default empty string, so if left blank, no extraction occurs.
    key: str = InputField(
        description="Metadata key to extract values for Output. Leave empty to ignore.",
        title="Metadata Key",
        default="",
        ui_order=1,
    )

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        """
        The core logic of the node.
        It iterates through the input image collection, fetches each image's metadata,
        and extracts values for the specified keys, compiling them into output lists.
        """

        # Initialize list to store extracted values for the provided key.
        collected_values: list[str] = []

        # Iterate through each ImageField object in the input 'images' collection.
        # ImageField objects contain the 'image_name' reference needed to load the metadata.
        for img_field in self.images:
            try:
                # Extract the metadata dictionary from the image field.
                # metadata.root is dict[str, Any].
                metadata: Dict[str, Any] = {}
                image_metadata = context.images.get_metadata(img_field.image_name)
                if image_metadata is not None:
                    metadata.update(image_metadata.root)

                # If no metadata is found for an image, log a warning and treat it as an empty dictionary
                # to ensure all output lists maintain consistent lengths.
                if not metadata:
                    warning(
                        f"No metadata found for image: '{img_field.image_name}'. Appending empty string."
                    )
                    metadata = {}  # Use an empty dictionary to avoid KeyError

                if self.key:  # Only attempt to extract if the input self.key string is not empty
                    # Use .get() with a default of "" to gracefully handle missing self.key
                    extracted_value = metadata.get(self.key, "")
                    collected_values.append(
                        str(extracted_value)
                    )  # Append the value, ensuring it's a string
                else:
                    # If the self.key input was empty, append an empty string to the output list.
                    # This ensures the output list has the same number of elements as the input image collection.
                    collected_values.append("")

            except Exception as e:
                # Catch any exceptions during image processing (e.g., image_name not found, metadata parsing errors).
                # Log the error and append an empty string for the current image.
                error(f"Error processing image '{img_field.image_name}': {e}")
                collected_values.append("")

        # Construct and return the output object.
        return StringCollectionOutput(collection=collected_values)
