import logging

class Application:
    """
    Class which implements the application logic for the GraphRAG pipeline.
    """

    def __init__(self, name: str):
        self.name = name
        self.__init_logging()
 
    def run(self):
        print(f"Running application: {self.name}")
    
    def __init_logging(self):
        """
        Initializes logging for the application.
        """

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )