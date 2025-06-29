import time

def get_rate_limit_checker(max_requests_per_minute: int):
    """
    Returns a function that checks and enforces a rate limit. Example usage:

    ```python
    rate_limit_checker = get_rate_limit_checker(60)  # 60 requests per minute
    for _ in range(100):
        rate_limit_checker()  # Call this before each request to check the rate limit
    ```
    This function will pause execution if the rate limit is reached, ensuring that no more than `max_requests_per_minute` requests are made in any one minute period.

    Args:
        max_requests_per_minute (int): The maximum number of requests allowed per minute.

    """
    request_count = 0
    start_time = time.time()

    def check_and_wait():

        nonlocal request_count, start_time  # Use nonlocal to modify outer scope variables

        # This function will be called before an LLM call, so we can increment here.
        request_count += 1
        
        print(f"Current request count: {request_count}. Time since last reset: {time.time() - start_time:.2f} seconds.")

        if request_count >= max_requests_per_minute:
            
            elapsed_time = time.time() - start_time
            
            if elapsed_time < 60:  # Add a delay of a bit more than one minute
                sleep_duration = 60 - elapsed_time
                print(f"Rate limit of {max_requests_per_minute} requests/minute reached. Pausing for {sleep_duration:.2f} seconds.")
                time.sleep(sleep_duration)
            
            # Reset counter and timer for the new minute window
            request_count = 0
            start_time = time.time()

    return check_and_wait