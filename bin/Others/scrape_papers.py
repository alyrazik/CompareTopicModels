import logging
import sys
import pandas as pd


# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('Network logger')
SEARCH_LIMIT = 3000   # limit the number of returned articles matching a keyword search


def scrape_save_to_cloud(mongo_client, database, table, group_ID, n_pages):
    """
    Scrapes a facebook group and saves the scraped content to a MongoDB

    :param mongo_client: the client name of a cloud MongoDB
    :param database: the name of the database to query
    :param table: the name of the table to query from.
    :param group_ID: the facebook group ID to scrape data from.
    :param n_pages: the number of pages to scrape from the group
    :return: none

    """
    try:

        client = MongoClient(mongo_client)

        # create a database
        db = client[database]
        table = db[table]

        posts_list = get_fb_posts(group=group_ID, pages=n_pages)

        df = pd.DataFrame(posts_list)
        df.to_csv('out.csv', encoding='UTF-8')

        if posts_list:
            table.delete_many({})
            table.insert_many(posts_list)
        else:
            raise Exception("No posts found.")

    except Exception:
        logger.exception('Exception occurred in scraping content from facebook')


def obtain_from_cloud(mongo_client, database, table):
    """
    Obtain data from MongoDB and saves it to a pandas data freme
    :param mongo_client: the client name of a cloud MongoDB
    :param database: the name of the database to query
    :param table: the name of the table to query from.
    :return: returns a dataframe with data.

    """
    try:
        # connect to MongoDB client
        client = MongoClient(mongo_client)
        db = client[database]
        df = retrieve_documents(database=db, collection=table)
        df.drop(['post_text', 'shared_text', 'image', 'video', 'video_thumbnail', 'video_id', 'images'], axis=1,
                inplace=True)

        return df

    except Exception:
        logger.exception('Exception occurred in obtaining documents from MongoDB')


def obtain_kw_from_cloud(search_text, mongo_client, database, table):
    """
      Take a a string containing keywords and outputs all relevant posts to any of them
      Args:
        -search_text: a string of keywords (<string>)
        -mongo_client: the client name of a cloud MongoDB
        -database: MongoDB database name (<pymongo.database.Database>)
        -collection: string with the name of MongoDB collection (<string>)
      Returns:
        -A pandas dataframe containing the MongoDB contents of the returned news articles
    """
    try:

        # connect to MongoDB client
        client = MongoClient(mongo_client)
        db = client[database]
        db[table].create_index([
            ("text", "text")
        ],
            name="search_index"
        )

        returned_cursor = db[table].find({"$text": {"$search": search_text}}).limit(SEARCH_LIMIT)
        df = pd.DataFrame(returned_cursor)
        return df

    except Exception:
        logger.exception('Exception occurred in obtaining searched keywords from MongoDB')
