from com.hansiz.bot.util.logger import logger
from com.hansiz.bot.util.pineconeutil import init


def main_init():
    """
    Main initialization function.
    """
    # Initialize Pinecone
    index = init()

    # Additional initializations can be added here
    logger.info("Initialization complete.")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_init()
