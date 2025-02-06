import os

import osmnx as ox
import pandas as pd

from config.paths import DATA_FILENAME, TRAVEL_TIMES_CSV
from config.settings import DEFAULT_NETWORK_TYPE


def download_and_save_graph(place_name="Kaunas, Lithuania", filename=DATA_FILENAME):
    """
    Downloads OSM data for the given place, adds speeds and travel times,
    and saves the graph as GraphML.
    """
    G = ox.graph_from_place(place_name, network_type=DEFAULT_NETWORK_TYPE)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    ox.save_graphml(G, filename)
    return True


def load_graph(filename=DATA_FILENAME):
    """
    Load the graph from GraphML if it exists. Raises an error if something goes wrong.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Graph data file not found at {filename}")
    G = ox.load_graphml(filename)
    return G


def update_travel_times_from_csv(G):
    """
    Read adjusted travel times from CSV and update the graph's edges.
    """
    try:
        adjusted_data = pd.read_csv(TRAVEL_TIMES_CSV)
        node_cache = {}

        def get_node(lat, lon):
            if (lat, lon) not in node_cache:
                node_cache[(lat, lon)] = ox.nearest_nodes(G, X=lon, Y=lat)
            return node_cache[(lat, lon)]

        for _, row in adjusted_data.iterrows():
            origin = (row["origin_lat"], row["origin_lon"])
            destination = (row["destination_lat"], row["destination_lon"])
            travel_time = row["travel_time_minutes"] * 60

            orig_node = get_node(*origin)
            dest_node = get_node(*destination)

            if G.has_edge(orig_node, dest_node):
                edge_data = G[orig_node][dest_node]
                if isinstance(edge_data, dict):
                    for key in edge_data.keys():
                        edge_data[key]["travel_time"] = travel_time
            else:
                print(f"Warning: Edge from {origin} to {destination} not found in graph.")

        print("Travel times updated successfully from CSV.")
    except Exception as e:
        print(f"Error updating travel times: {e}")
