"""
This module contains functions for saving dialogues to a CosmoDB database.
"""

import os

import pymongo

from pipelines.dialogue_state import DialogueState
from utils.logging import logger

cosmos_db_client = None
dialogue_db_collection = None
preference_db_collection = None


def initialize_db_connection():
    logger.info("Initializing Cosmos DB connection")
    CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
    if not CONNECTION_STRING:
        raise ValueError("COSMOS_CONNECTION_STRING environment variable not set")

    global cosmos_db_client, dialogue_db_collection, preference_db_collection
    cosmos_db_client = pymongo.MongoClient(CONNECTION_STRING)
    db = cosmos_db_client["wikichat"]  # the database name is wikichat
    dialogue_db_collection = db[
        "dialog_turns"
    ]  # the collection that stores dialog turns and their user ratings
    dialogue_db_collection.create_index(
        "$**"
    )  # necessary to build an index before we can call sort()
    preference_db_collection = db[
        "preferences"
    ]  # the collection that stores information about what utterance users preferred
    preference_db_collection.create_index(
        "$**"
    )  # necessary to build an index before we can call sort()
    # The "schema" of dialogue_db_collection is: {_id=(dialog_id, turn_id, system_name), experiment_id, dialog_id, turn_id, system_name, user_utterance, agent_utterance, agent_log_object, user_naturalness_rating}
    # The "schema" of preference_db_collection is: {_id=(dialog_id, turn_id), experiment_id, dialog_id, turn_id, winner_system, loser_systems}


def save_dialogue_to_db(
    dialogue_state: DialogueState,
    dialogue_id: str,
    system_name: str,
    experiment_id: str = "default-experiment",
):
    if cosmos_db_client is None:
        initialize_db_connection()
    entries_to_write = []
    for turn_id, dialogue_turn in enumerate(dialogue_state.turns):
        entries_to_write.append(
            {
                "_id": str((dialogue_id, turn_id, system_name)),
                "experiment_id": experiment_id,
                "dialog_id": dialogue_id,
                "turn_id": turn_id,
                "system_name": system_name,
                "user_utterance": dialogue_turn.user_utterance,
                "agent_utterance": dialogue_turn.agent_utterance,
                "agent_log_object": dialogue_turn.model_dump(),
            }
        )
    try:
        logger.info(f"Saving dialogue '{dialogue_id}' to database")
        dialogue_db_collection.insert_many(entries_to_write)
    except Exception as e:
        logger.error(f"Could not save '{dialogue_id}' to database: {e}")
