def format_message(name):
    """Example function to format a personalized 'Hello World' message

    :param name: the name to use for personalization
    :type name: str
    :return: str, a perfonalized 'Hello World' message
    """
    if not name:
        raise ValueError("Name must be a valid name")
    return "Hello {}".format(name.capitalize())
