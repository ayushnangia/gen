# dialogue_generation/utils/time_generator.py
"""
Time generation utility extracted from serial_gen.py and parallel_gen.py
EXACT extraction with no modifications to preserve functionality
"""

import random
import logging
from typing import Tuple


class TimeGenerator:
    """
    Time generation utility for generating random times within time slots
    Extracted from generate_random_time method in both DialogueGenerator classes
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def generate_random_time(self, time_slot: Tuple[int, int, str]) -> str:
        """
        Generates a random time within a given time slot in 5-minute intervals.
        Handles overnight slots correctly and ensures uniform distribution.

        EXACT copy from serial_gen.py lines 291-344 and parallel_gen.py lines 313-366

        Args:
            time_slot: Tuple of (start_hour, end_hour, slot_name)

        Returns:
            String representation of the random time (HH:MM format)
        """
        try:
            start_hour, end_hour = time_slot[0], time_slot[1]

            # Validate input hours
            if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
                self.logger.error(f"Invalid hours in time slot: {time_slot}. Using default time.")
                return "00:00"

            # Handle overnight slots (e.g., 23:00 to 05:00)
            if end_hour < start_hour:
                if random.random() < 0.5:
                    # Night portion (start_hour to midnight)
                    start_minutes = start_hour * 60
                    end_minutes = 24 * 60  # midnight
                else:
                    # Morning portion (midnight to end_hour)
                    start_minutes = 0
                    end_minutes = end_hour * 60
            else:
                # Normal daytime slot
                start_minutes = start_hour * 60
                end_minutes = end_hour * 60

            # Generate random minutes within the range
            total_minutes = random.randint(start_minutes, end_minutes - 1)

            # Convert to hours and minutes
            hours = (total_minutes // 60) % 24
            mins = total_minutes % 60

            # Round to nearest 5 minutes
            mins = round(mins / 5) * 5
            if mins == 60:
                hours = (hours + 1) % 24
                mins = 0

            generated_time = f"{hours:02d}:{mins:02d}"
            self.logger.debug(f"Generated time {generated_time} for slot {time_slot[2]}")
            return generated_time

        except Exception as e:
            self.logger.error(f"Error generating random time for slot {time_slot}: {str(e)}")
            return "00:00"  # Return default time in case of any error


class SecureTimeGenerator(TimeGenerator):
    """
    Secure version using cryptographically secure random
    Based on parallel_gen.py implementation with SystemRandom
    """

    def __init__(self, logger=None):
        super().__init__(logger)
        self._system_random = random.SystemRandom()

    def generate_random_time(self, time_slot: Tuple[int, int, str]) -> str:
        """
        Same as base implementation but using cryptographically secure random
        EXACT copy from parallel_gen.py with SystemRandom usage
        """
        try:
            start_hour, end_hour = time_slot[0], time_slot[1]

            # Validate input hours
            if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
                self.logger.error(f"Invalid hours in time slot: {time_slot}. Using default time.")
                return "00:00"

            # Handle overnight slots (e.g., 23:00 to 05:00)
            if end_hour < start_hour:
                if self._system_random.random() < 0.5:
                    # Night portion (start_hour to midnight)
                    start_minutes = start_hour * 60
                    end_minutes = 24 * 60  # midnight
                else:
                    # Morning portion (midnight to end_hour)
                    start_minutes = 0
                    end_minutes = end_hour * 60
            else:
                # Normal daytime slot
                start_minutes = start_hour * 60
                end_minutes = end_hour * 60

            # Generate random minutes within the range using secure random
            total_minutes = self._system_random.randint(start_minutes, end_minutes - 1)

            # Convert to hours and minutes
            hours = (total_minutes // 60) % 24
            mins = total_minutes % 60

            # Round to nearest 5 minutes
            mins = round(mins / 5) * 5
            if mins == 60:
                hours = (hours + 1) % 24
                mins = 0

            generated_time = f"{hours:02d}:{mins:02d}"
            self.logger.debug(f"Generated time {generated_time} for slot {time_slot[2]}")
            return generated_time

        except Exception as e:
            self.logger.error(f"Error generating random time for slot {time_slot}: {str(e)}")
            return "00:00"  # Return default time in case of any error