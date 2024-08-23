"""
The backend API that runs dialog agents, returns agent utterance to the front-end, and stores user data in a MongoDB database

The API has the following three functions that can be used by any front-end.
All inputs/outputs are string, except for `log_object` which is a json object and `turn_id` and `user_naturalness_rating` which are integers.
- `/chat`
Inputs: (experiment_id, new_user_utterance, dialog_id, turn_id, system_name)
Outputs: (agent_utterance, log_object)
Each time a user types something and clicks send, the front-end should make one call per system to /chat. So e.g. it should make two separate calls for two systems.

- `/user_rating`
Inputs: (experiment_id, dialog_id, turn_id, system_name, user_naturalness_rating, user_factuality_rating, user_factuality_confidence)
Outputs: None
When the user submits their ratings, the front-end should make one call per system to /user_rating. So e.g. it should make two separate calls for two systems.

- `/user_preference`
Inputs: (experiment_id, dialog_id, turn_id, winner_system, loser_systems)
Outputs: None
Each time the user selects one of the agent utterances over the other, you make one call to /user_preference.

`turn_id` starts from 0 and is incremented by 1 after a user and agent turn
"""

import os
import warnings

from pipelines.dialogue_state import DialogueState

warnings.filterwarnings(
    "ignore", "You appear to be connected to a CosmosDB cluster", category=UserWarning
)


import pymongo

from pipelines.utils import get_logger

logger = get_logger(__name__)

# set up the MongoDB connection
CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
client = pymongo.MongoClient(CONNECTION_STRING)
db = client["wikichat"]  # the database name is wikichat
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
    entries_to_write = []
    for turn_id, dialogue_turn in enumerate(dialogue_state["dialogue_history"]):
        entries_to_write.append(
            {
                "_id": str((dialogue_id, turn_id, system_name)),
                "experiment_id": experiment_id,
                "dialog_id": dialogue_id,
                "turn_id": turn_id,
                "system_name": system_name,
                "user_utterance": dialogue_turn.user_utterance,
                "agent_utterance": dialogue_turn.agent_utterance,
                "agent_log_object": dialogue_turn.turn_log,
                "user_naturalness_rating": -1,
                "user_factuality_rating": False,
                "user_factuality_confidence": -1,
            }
        )
    try:
        logger.info(
            "Inserting new dialogue with dialogue_id '%s' to database",
            str(dialogue_id),
        )
        dialogue_db_collection.insert_many(entries_to_write)
    except Exception as e:
        logger.error("Could not save to database")
        logger.exception(e)
