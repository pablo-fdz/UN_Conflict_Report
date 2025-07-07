import time

def get_rate_limit_checker(max_requests_per_minute: int, max_tokens_per_minute: int = 250000):
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
        max_tokens_per_minute (int): The maximum number of tokens allowed per minute. Default is 250,000.

    """
    request_count = 0
    token_count = 0
    start_time = time.time()

    def check_and_wait(tokens_used: int = 0):

        nonlocal request_count, token_count, start_time  # Use nonlocal to modify outer scope variables

        # This function will be called before an LLM call, so we can increment here.
        request_count += 1
        token_count += tokens_used
        
        elapsed_time = time.time() - start_time
        
        print(f"Current request count: {request_count}/{max_requests_per_minute}. "
              f"Token count: {token_count}/{max_tokens_per_minute}. "
              f"Time since last reset: {elapsed_time:.2f} seconds.")

        request_limit_reached = request_count >= max_requests_per_minute
        token_limit_reached = token_count >= max_tokens_per_minute
        
        if request_limit_reached or token_limit_reached:
            if elapsed_time < 60: 
                sleep_duration = 60 - elapsed_time + 1  # 1 second buffer
            
                if request_limit_reached and token_limit_reached:
                    print(f"Both request limit ({max_requests_per_minute}) and token limit ({max_tokens_per_minute}) reached. "
                          f"Pausing for {sleep_duration:.2f} seconds.")
                elif request_limit_reached:
                    print(f"Request limit of {max_requests_per_minute} requests/minute reached. "
                          f"Pausing for {sleep_duration:.2f} seconds.")
                else:
                    print(f"Token limit of {max_tokens_per_minute} tokens/minute reached. "
                          f"Pausing for {sleep_duration:.2f} seconds.")
                
                time.sleep(sleep_duration)

            # Reset counters and timer for the new minute window
            request_count = 0
            token_count = 0
            start_time = time.time()

    return check_and_wait