import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

def date_range_converter(range_statement: str) -> tuple[str, str]:
    """
    Converts a human-readable date range statement into a start and end date.

    The range is calculated backwards from the current date.

    Args:
        range_statement (str): A string like "1 day", "4 weeks", "2 months", "1 year".
            Accepts singular and plural forms of the following time units: day, week, month, year.

    Returns:
        tuple: A tuple containing the start date and end date in 'YYYY-MM-DD' format.
    
    Raises:
        ValueError: If the range_statement is in an invalid format.
    """
    end_date = datetime.now().date()

    match = re.match(r"(\d+)\s+(day|week|month|year)s?", range_statement, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid range statement: '{range_statement}'. Expected format like 'X days/weeks/months/years'.")

    quantity = int(match.group(1))
    unit = match.group(2).lower()

    if unit == "day":
        delta = relativedelta(days=quantity)
    elif unit == "week":
        delta = relativedelta(weeks=quantity)
    elif unit == "month":
        delta = relativedelta(months=quantity)
    elif unit == "year":
        delta = relativedelta(years=quantity)
    
    start_date = end_date - delta
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')