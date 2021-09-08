import unittest

from {{cookiecutter._internal_vars.package_name}} import example


class TestFormatMessage(unittest.TestCase):
    def test_format_message_invalid_name(self):
        """Test behaviour of message formatting, with an invalid name"""
        with self.assertRaises(ValueError):
            example.format_message(None)

    def test_format_message_valid_name(self):
        """Test behaviour of message formatting, with a valid name"""
        result = example.format_message("james")
        self.assertEqual(result, "Hello James")
