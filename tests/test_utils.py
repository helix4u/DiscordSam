import unittest
from datetime import datetime, timezone

from utils import append_absolute_dates


class AppendAbsoluteDatesTest(unittest.TestCase):
    def test_relative_phrase(self) -> None:
        base_time = datetime(2024, 5, 8, tzinfo=timezone.utc)
        text = "Let's meet tomorrow."
        result = append_absolute_dates(text, current_time=base_time)
        self.assertIn("tomorrow (2024-05-09 00:00 UTC)", result)

    def test_absolute_date_unchanged(self) -> None:
        text = "Tell me (2025-08-05 18:00 Mountain Daylight Time) about gogurtius."
        result = append_absolute_dates(text)
        self.assertEqual(result, text)


if __name__ == "__main__":
    unittest.main()
