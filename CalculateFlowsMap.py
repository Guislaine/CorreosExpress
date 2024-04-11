"""Minimum cost flow supply/demand et carte """

def import_data():
 
    import numpy as np
    import pandas as pd
     
    # Load excel file in DataFrame
    df = pd.read_excel('https://github.com/Guislaine/CorreosExpress/raw/main/NetworkCEX.xlsx')
    df_start_nodes = df['start_nodes'].tolist()
    df_end_nodes = df['end_nodes'].tolist()
    df_capacities = df['capacities'].tolist()
    df_unit_costs = df['costs'].tolist()
    
    #df = pd.read_excel('/Users/guislainegall/Downloads/SuppliesCEX.xlsx')
    #df_supplies = df['supplies'].tolist()
    
    # Define five parallel arrays: sources, destinations, capacities,
    # and unit costs between each pair + supplies at each node
    start_nodes = np.array(df_start_nodes)
    end_nodes = np.array(df_end_nodes)
    capacities = np.array(df_capacities)
    unit_costs = np.array(df_unit_costs)
    
    # Faire la liste des delegacion de origen
    list_del = sorted(df['start_del'].unique())
#    list_del = ['MADRID CTC']
    
    # Copier dans df les données de nombre de camion et de cout du fichier cost per flow 
    df_costperflow = pd.read_excel('https://github.com/Guislaine/CorreosExpress/raw/main/cost%20per%20flow.xlsx')
    
    
    return df, start_nodes, end_nodes, capacities, unit_costs, list_del, df_costperflow
    
        
def calculate_flows(df, value):

    import numpy as np
    import pandas as pd
    from ortools.graph.python import min_cost_flow
    import sys
    
    # Instantiate a SimpleMinCostFlow solver.
    smcf = min_cost_flow.SimpleMinCostFlow()
    
    # Créer une copie de df etmettre à 0 la capacité des lignes
#    value = 'BARCELONA'
    df_by_origen = df.copy()
    df_by_origen.loc[df_by_origen['start_del'] != value, 'envios'] = 0
    df_by_origen.loc[df_by_origen['start_del'] != value, 'percibidos'] = 0


    # Utilisez la méthode loc pour mettre à zéro les valeurs de "envios" et "percibidos" lorsque "in_demand" est égal à 0
#    df_by_origen.loc[df_by_origen['in_demand'] == 0, ['envios', 'percibidos']] = 0

    # Regrouper par le nom des villes de départ
    start_grouped = df_by_origen.groupby(['start_del', 'start_nodes'])[['envios']].sum()
    start_grouped.reset_index(inplace=True)
    start_grouped.columns = ['nombre_del', 'nodes', 'supplies']
    
    # Regrouper par le nom des villes d'arrivée
    end_grouped = df_by_origen.groupby(['end_del', 'end_nodes'])[['percibidos']].sum()
    end_grouped.reset_index(inplace=True)
    end_grouped.columns = ['nombre_del', 'nodes', 'supplies']
    
    # Combiner les deux groupes en utilisant l'index des villes
    df_supplies = pd.concat([start_grouped, end_grouped])
    df_supplies = df_supplies.groupby(['nombre_del', 'nodes'])[['supplies']].sum()

    
    # Calculer la somme de la colonne "supplies" de "combined_grouped"
    total_supplies = df_supplies['supplies'].sum()

    
    # Afficher la somme totale
    print("Somme totale des supplies:", total_supplies)

    # Define an array of supplies at each node.
    supplies = np.array(df_supplies['supplies'])


    # Define an array of supplies at each node.
    #supplies = np.array(df_supplies)

    # Add arcs, capacities and costs in bulk using numpy.
    all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
        start_nodes, end_nodes, capacities, unit_costs
    )

    # Add supply for each nodes.
    smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)

    # Find the min cost flow.
    status = smcf.solve()

    if status != smcf.OPTIMAL:
        print("There was an issue with the min cost flow input.")
        print(f"Status: {status}")
        sys.exit(1)
    else:
        print(value, ":OK")
           
    # Create a DataFrame to store the results
    results_df = pd.DataFrame(columns=["start_del", "end_del", "flow_opt", "costs"])
    
    solution_flows = smcf.flows(all_arcs)
    costs = solution_flows * unit_costs
    results_dfs = []
    for arc, flow, cost in zip(all_arcs, solution_flows, costs):
        start_del = smcf.tail(arc)
        end_del = smcf.head(arc)
        results_dfs.append(pd.DataFrame({"start_del": [start_del], "end_del": [end_del], "flow_opt": [flow], "costs": [cost]}))

    results_df = pd.concat(results_dfs, ignore_index=True)
 
    #for arc, flow, cost in zip(all_arcs, solution_flows, costs):
        #start_del = smcf.tail(arc)
        #end_del = smcf.head(arc)
        #results_df = results_df.append({"start_del": start_del, "end_del": end_del, "flow_opt": flow, "costs": cost}, ignore_index=True)
    
    df['flow_opt'] += results_df['flow_opt']
    
    return df


def calculate_total_cost(df, df_costperflow):
    
    import numpy as np
    
    # Inclure dans df les données de cost per flow
    for index_df in range(len(df)):
        index_df_costperflow = df_costperflow[(df_costperflow['flow_min'] < df['flow_opt'][index_df]) & (df_costperflow['flow_max'] >= df['flow_opt'][index_df])].index[0]
        df.loc[index_df, 'nb_furgon'] = df_costperflow.loc[index_df_costperflow, 'nb_furgon']
        df.loc[index_df, 'nb_carrozado'] = df_costperflow.loc[index_df_costperflow, 'nb_carrozado']
        df.loc[index_df, 'nb_rigido'] = df_costperflow.loc[index_df_costperflow, 'nb_rigido']
        df.loc[index_df, 'nb_trailer'] = df_costperflow.loc[index_df_costperflow, 'nb_trailer']
        df.loc[index_df, 'arc_cost_km'] = df_costperflow.loc[index_df_costperflow, 'arc_cost_km']
        df.loc[index_df, 'color'] = df_costperflow.loc[index_df_costperflow, 'color']
        df.loc[index_df, 'arc_cost'] = df.loc[index_df, 'arc_cost_km'] * df.loc[index_df, 'distancia'] 
        
    # Calculer le cout total
    total_cost = df['arc_cost'].sum()
    print("Total cost: ", np.round(total_cost),"€")
    
    # Save the results to an Excel file
    print(df.head())
    #df.to_excel("https://github.com/Guislaine/CorreosExpress/raw/main/MinCostFlowResults.xlsx", index=False)
    
    return df
        
  
def create_map(df):
    
    import networkx as nx
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import pandas as pd
    
    # Create an empty directed graph
    G = nx.DiGraph()
    
    # Add cities as nodes with their geographical coordinates
    # Read the Excel file
    df_del = pd.read_excel("https://github.com/Guislaine/CorreosExpress/raw/main/delegacion.xlsx")
    df_del['latitud'] = df_del['latitud'].str.replace(',', '.').astype(float)
    df_del['longitud'] = df_del['longitud'].str.replace(',', '.').astype(float)
    
    # Initialize an empty dictionary
    villes = {}
    
    # Iterate over the rows of the DataFrame to extract data
    for index, row in df_del.iterrows():
        nom_ville = row["del"]
        latitude = row["latitud"]
        longitude = row["longitud"]
    
        # Add the data to the dictionary
        villes[nom_ville] = (longitude, latitude)
    
    for ville, coords in villes.items():
        G.add_node(ville, pos=coords)
    
    # Add directed edges to connect cities with the "flow" and "color" attributes
    # Read the Excel file
    # df = pd.read_excel("https://github.com/Guislaine/CorreosExpress/raw/main/edge.xlsx")
    
    # Create a dictionary to map color names to actual colors
    color_mapping = {
        "red": "red",
        "orange": "orange",
        "yellow": "yellow",
        "green": "green",
        # Add more colors as needed
    }
    
    # Select only the edges with a non null flow
    df_results = df.copy()
    df_results = df_results.loc[df_results['flow_opt'] != 0]
    
    # Add the "flow" attribute and specify edge colors
    for index, row in df_results.iterrows():
        start_city = row["start_del"]
        end_city = row["end_del"]
        flow = row["flow_opt"]
        color = color_mapping.get(row["color"], "gray")  # Default to gray if the color is not recognized
        
        G.add_edge(start_city, end_city, flow=flow, color=color)
    
    # Set the figure size with higher resolution (e.g., 1920x1080)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 8), dpi=300)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 8), dpi=300)
    ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())
    
    # Add the borders of Spain and Portugal
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    
    # Display the directed graph in overlay
    node_positions = {ville: coords for ville, coords in nx.get_node_attributes(G, 'pos').items()}
    edge_colors = [color_mapping.get(G[start_city][end_city]['color'], "gray") for start_city, end_city in G.edges()]
    nx.draw_networkx_nodes(G, node_positions, node_size=25, node_color='skyblue')
    nx.draw_networkx_edges(G, node_positions, width=1, alpha=0.5, edge_color=edge_colors, arrows=True)
    
    # Label the nodes with city names only on the graph
    labels = {ville: ville for ville in G.nodes()}
    nx.draw_networkx_labels(G, node_positions, labels, font_size=4, font_color='black')
    
    # Display the labels of the "flow" attribute next to the edges
    edge_labels = nx.get_edge_attributes(G, 'flow')  # Get the values of the "flow" attribute
    nx.draw_networkx_edge_labels(G, node_positions, edge_labels=edge_labels, font_size=3)
    
    # Display the legend
 #   plt.legend(["Villes"])
    
    # Display the map
    plt.title("Mapo de flujos")
    plt.show()
    

if __name__ == "__main__":
    df, start_nodes, end_nodes, capacities, unit_costs, list_del, df_costperflow = import_data()
    for value in list_del:
        df = calculate_flows(df, value)
    calculate_total_cost(df, df_costperflow)
    create_map(df)
