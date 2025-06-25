import sqlite3
import uuid
import io
import os
import time
import math # Import math for ceil function

import torch
from typing import Optional, Union

from invokeai.invocation_api import (
    BaseInvocation,
    InputField,
    InvocationContext,
    invocation,
    OutputField,
    invocation_output,
    BaseInvocationOutput,
    StringOutput,
)
from invokeai.app.invocations.fields import (
    FluxConditioningField,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    FLUXConditioningInfo,
    ConditioningFieldData,
)
from invokeai.backend.util.logging import info, warning, error

# Define the database file name
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flux_conditionings.db")

# Global variables for database size management
MAX_DB_SIZE_MB = 1024
MARGIN_DB_SIZE_MB = 192
WARNING_THRESHOLDS_MB = [96, 48, 24, 12] # Warning thresholds below the margin

# Convert to bytes
MAX_DB_SIZE_BYTES = MAX_DB_SIZE_MB * 1024 * 1024
MARGIN_DB_SIZE_BYTES = MARGIN_DB_SIZE_MB * 1024 * 1024


def _get_db_size():
    """Returns the current size of the database file in bytes."""
    if os.path.exists(DB_FILE):
        return os.path.getsize(DB_FILE)
    return 0


def _manage_db_size():
    """
    Manages the database size, deleting a fraction of oldest entries if the maximum size is exceeded.
    Logs warnings at predefined thresholds and VACUUMs the database after deletion.
    """
    current_size = _get_db_size()
    info(f"Current database size: {current_size / (1024 * 1024):.2f} MB")

    # Check for warnings based on remaining space before MAX_DB_SIZE_BYTES
    remaining_space_to_max = MAX_DB_SIZE_BYTES - current_size
    for threshold_mb in WARNING_THRESHOLDS_MB:
        threshold_bytes = threshold_mb * 1024 * 1024
        # Warn if remaining space falls below a threshold and is still above the next lower threshold
        # This prevents repeated warnings for the same threshold level
        if remaining_space_to_max < threshold_bytes and (remaining_space_to_max >= (threshold_bytes / 2) or threshold_mb == WARNING_THRESHOLDS_MB[-1]):
            warning(f"Database space critically low! Only {remaining_space_to_max / (1024 * 1024):.2f} MB remaining until max. Current size: {current_size / (1024 * 1024):.2f} MB")
            break # Only trigger the highest applicable warning

    if current_size > MAX_DB_SIZE_BYTES:
        info(f"Database size {current_size / (1024 * 1024):.2f} MB exceeds maximum allowed size of {MAX_DB_SIZE_MB} MB. Trimming oldest entries.")
        conn = None
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()

            # Calculate the number of entries to delete
            # We want to reduce the size by at least MARGIN_DB_SIZE_BYTES
            # Estimate the average size per entry (simplistic but works for deletion ratio)
            cursor.execute("SELECT COUNT(*) FROM flux_conditionings")
            total_entries = cursor.fetchone()[0]

            if total_entries == 0:
                info("Database is empty, no entries to delete.")
                return

            # Calculate the target size to reduce to (MAX_DB_SIZE_BYTES - MARGIN_DB_SIZE_BYTES)
            # And the amount of bytes to delete to reach that target, plus a bit more for margin
            bytes_to_delete = current_size - (MAX_DB_SIZE_BYTES - MARGIN_DB_SIZE_BYTES)
            
            # If for some reason bytes_to_delete is negative (e.g., current_size is already within margin)
            # or if it's too small, ensure we delete at least a minimum fraction or a few entries.
            if bytes_to_delete <= 0:
                 # If current size is already within target, delete a small, fixed percentage to stay ahead
                delete_ratio = MARGIN_DB_SIZE_BYTES / MAX_DB_SIZE_BYTES
                num_entries_to_delete = math.ceil(total_entries * delete_ratio)
                info(f"Database is within target size, proactively deleting {num_entries_to_delete} entries.")
            else:
                # Calculate ratio based on current overflow and total size to determine deletion percentage
                delete_ratio = bytes_to_delete / current_size
                num_entries_to_delete = math.ceil(total_entries * delete_ratio)
                info(f"Calculated to delete {num_entries_to_delete} entries based on size overflow.")

            # Ensure we delete at least 1 entry if there's an overflow
            num_entries_to_delete = max(1, num_entries_to_delete)
            
            info(f"Attempting to delete {num_entries_to_delete} oldest entries.")

            # Delete the oldest entries
            cursor.execute(
                "DELETE FROM flux_conditionings WHERE id IN (SELECT id FROM flux_conditionings ORDER BY timestamp ASC LIMIT ?)",
                (num_entries_to_delete,)
            )
            conn.commit()
            info(f"Deleted {num_entries_to_delete} oldest entries.")

            # VACUUM the database to reclaim space
            info("Vacuuming database to reclaim space...")
            cursor.execute("VACUUM")
            conn.commit()
            info(f"Database vacuumed. New size: {_get_db_size() / (1024 * 1024):.2f} MB")

        except sqlite3.Error as e:
            error(f"Error managing database size: {e}")
        finally:
            if conn:
                conn.close()
    elif current_size > (MAX_DB_SIZE_BYTES - MARGIN_DB_SIZE_BYTES):
        warning(f"Database size {current_size / (1024 * 1024):.2f} MB is within the margin of {MARGIN_DB_SIZE_MB} MB from maximum size. Consider reviewing usage.")


# Create the SQLite database and table if they don't exist
def _init_db():
    """Initializes the SQLite database and creates the table for storing conditioning data."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS flux_conditionings (
                id TEXT PRIMARY KEY,
                clip_embeds BLOB NOT NULL,
                t5_embeds BLOB NOT NULL,
                timestamp REAL NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()
        info(f"SQLite database '{DB_FILE}' initialized successfully.")
    except sqlite3.Error as e:
        error(f"Error initializing database: {e}")

# Call the initialization function when the module is loaded
_init_db()

@invocation_output("flux_conditioning_store_output")
class FluxConditioningStoreOutput(BaseInvocationOutput):
    """Output for the Store Flux Conditioning node."""
    conditioning_id: str = OutputField(description="Unique identifier for the stored Flux Conditioning")


@invocation(
    "store_flux_conditioning",
    title="Store Flux Conditioning",
    tags=["conditioning", "flux", "database", "store"],
    category="conditioning",
    version="1.0.1", # Updated version due to functional change
    use_cache=False, # This node modifies external state (database), so caching should be off
)
class StoreFluxConditioningInvocation(BaseInvocation):
    """
    Stores a FLUX Conditioning object (CLIP and T5 embeddings) into an SQLite database.
    Returns a unique identifier for retrieval.
    Includes database size management with proactive deletion and VACUUM.
    """

    conditioning: FluxConditioningField = InputField(
        description="The FLUX Conditioning object to store.",
        ui_order=0,
    )

    def invoke(self, context: InvocationContext) -> FluxConditioningStoreOutput:
        """
        Main invocation method to store the conditioning.
        """
        # Load the FLUX Conditioning object
        try:
            loaded_cond = context.conditioning.load(self.conditioning.conditioning_name).conditionings[0]
            if not isinstance(loaded_cond, FLUXConditioningInfo):
                error("Input is not a valid FLUXConditioningInfo object.")
                return FluxConditioningStoreOutput(conditioning_id="")
        except Exception as e:
            error(f"Failed to load conditioning {self.conditioning.conditioning_name}: {e}")
            return FluxConditioningStoreOutput(conditioning_id="")

        clip_embeds_tensor = loaded_cond.clip_embeds
        t5_embeds_tensor = loaded_cond.t5_embeds

        if clip_embeds_tensor is None or t5_embeds_tensor is None:
            error("FLUX Conditioning object is missing CLIP or T5 embeddings.")
            return FluxConditioningStoreOutput(conditioning_id="")

        # Generate a unique identifier
        conditioning_id = str(uuid.uuid4())

        # Serialize tensors to bytes
        try:
            clip_buffer = io.BytesIO()
            torch.save(clip_embeds_tensor, clip_buffer)
            clip_bytes = clip_buffer.getvalue()

            t5_buffer = io.BytesIO()
            torch.save(t5_embeds_tensor, t5_buffer)
            t5_bytes = t5_buffer.getvalue()
        except Exception as e:
            error(f"Failed to serialize tensors: {e}")
            return FluxConditioningStoreOutput(conditioning_id="")

        # Store in SQLite
        conn = None
        try:
            _manage_db_size() # Manage size before inserting new data
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            timestamp = time.time() # Get current timestamp
            cursor.execute(
                "INSERT INTO flux_conditionings (id, clip_embeds, t5_embeds, timestamp) VALUES (?, ?, ?, ?)",
                (conditioning_id, clip_bytes, t5_bytes, timestamp),
            )
            conn.commit()
            info(f"Stored conditioning with ID: {conditioning_id}")
            return FluxConditioningStoreOutput(conditioning_id=conditioning_id)
        except sqlite3.Error as e:
            error(f"Error storing conditioning in database: {e}")
            return FluxConditioningStoreOutput(conditioning_id="")
        finally:
            if conn:
                conn.close()
