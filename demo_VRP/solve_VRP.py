import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

def main():
	folder = sys.argv[1]
	instance = load_instance_text(folder)
	
	#plot_instance(instance)

	#------------------------------
	#version sans capacité
	#------------------------------

	
	#start_time = time.time()
	
	sol_gurobi = gurobi_vrp_subtour_elim(instance)
	#end_time = time.time()
	plot_solution(instance, sol_gurobi)
	
	
	#print(f"Execution time: {end_time - start_time:.2f} seconds")

	sol_savings = savings_vrp(instance)
	#plot_solution(instance, sol_savings)
	
	sol_savings2 = savings_vrp_alt(instance)
	plot_solution(instance, sol_savings2)

def savings_vrp_alt(instance):
	"""
	Solves the VRP using the Savings Algorithm.

	Args:
		instance (dict): The VRP instance containing clients, depot, demands, vehicle capacity, etc.

	Returns:
		dict: Solution with arcs and total cost.
	"""
	depot_index = instance["num_clients"]  # Index of the depot
	clients = instance["clients"]
	demands = instance["demands"]
	veh_cap = instance["veh_cap"]
	vertices = np.vstack([clients, instance["depot"]])
	n_clients = len(clients)
	n_vertices = len(vertices)

	def dist(point1, point2):
		return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

	def createSol(routes):
		# Step 4: Convert routes to arcs
		arcs = []
		total_cost = 0
		for route in routes:
			route_arcs = []
			prev = depot_index
			for client in route:
				route_arcs.append((prev, client))
				total_cost += distances[prev][client]
				prev = client
			route_arcs.append((prev, depot_index))
			total_cost += distances[prev][depot_index]
			arcs.append(route_arcs)

		return {
			"arcs": arcs,
			"total_cost": total_cost
		}
	# Compute distance matrix
	distances = [[dist(vertices[i], vertices[j]) for j in range(n_vertices)] for i in range(n_vertices)]

	# Step 1: Initialize routes (one route per client)
	routes = [[i] for i in range(n_clients)]
	route_costs = [2 * distances[depot_index][i] for i in range(n_clients)]  # Cost of visiting each client directly
	route_demands = [demands[i] for i in range(n_clients)]

	# Step 2: Compute savings
	savings = []
	for i in range(n_clients):
		for j in range(n_clients):
			saving = distances[depot_index][i] + distances[depot_index][j] - distances[i][j]
			savings.append((saving, i, j))
	savings.sort(reverse=True, key=lambda x: x[0])  # Sort savings in descending order

	#print(savings)
	#exit(0)
	finishedRoutes = []

	currRouteMergable = True
	currRoute = None
	while(currRoute != None or currRouteMergable == True):
		
		if(currRoute != None and not currRouteMergable):
			print("removing ",currRoute)
			finishedRoutes.append(currRoute)
			routes.remove(currRoute)
			currRoute = None
		print("mergable")
		print(routes)
		
		currRouteMergable=False
		# Step 3: Merge routes based on savings
		for saving, i, j in savings:
			# Find routes containing i and j
			route_i = next((r for r in routes if i in r), None)
			route_j = next((r for r in routes if j in r), None)

			if(currRoute !=None and route_i != currRoute ):
				continue
			
			
				print(currRoute)
			# Check if merging is possible
			if route_i is not None and route_j is not None and route_i != route_j:
				if sum([demands[i] for i in route_i]) + sum([demands[i] for i in route_j]) <= veh_cap:
					print("somme des demandes : ",  sum([demands[i] for i in route_i]) + sum([demands[i] for i in route_j]) )
					#if(currRoute == None):
					#	currRoute = route_i
					print("route courante")
					print(route_i)
					
					print("merge avec ")
					print(route_j)
					canMerge=False
					# Merge routes
					if route_i[-1] == i and route_j[0] == j:
						canMerge = True
						merged_route = route_i + route_j
					elif route_i[0] == i and route_j[-1] == j:
						canMerge = True
						merged_route = route_j + route_i
					elif route_i[-1] == i and route_j[-1] == j:
						canMerge = True
						merged_route = route_i + route_j.reverse()
					elif route_i[0] == i and route_j[0] == i:
						canMerge = True
						merged_route = route_j.reverse() + route_i.reverse()
					else:
						continue
					
					if(canMerge):
						
						currRoute = merged_route
						currRouteMergable = True
						print(merged_route)
						# Update route demand and cost

						route_demands[routes.index(route_i)] += route_demands[routes.index(route_j)]
						route_costs[routes.index(route_i)] = (
							route_costs[routes.index(route_i)]
							+ route_costs[routes.index(route_j)]
							- saving
						)
						# Update the routes list
						routes.remove(route_i)
						routes.remove(route_j)
						routes.append(merged_route)
						sol = createSol(routes+finishedRoutes)		
						#plot_solution(instance, sol)
						break
		
	
	return createSol(routes+finishedRoutes)	

def savings_vrp(instance):
	"""
	Solves the VRP using the Savings Algorithm.

	Args:
		instance (dict): The VRP instance containing clients, depot, demands, vehicle capacity, etc.

	Returns:
		dict: Solution with arcs and total cost.
	"""
	depot_index = instance["num_clients"]  # Index of the depot
	clients = instance["clients"]
	demands = instance["demands"]
	veh_cap = instance["veh_cap"]
	vertices = np.vstack([clients, instance["depot"]])
	n_clients = len(clients)
	n_vertices = len(vertices)

	def dist(point1, point2):
		return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

	# Compute distance matrix
	distances = [[dist(vertices[i], vertices[j]) for j in range(n_vertices)] for i in range(n_vertices)]

	# Step 1: Initialize routes (one route per client)
	routes = [[i] for i in range(n_clients)]
	route_costs = [2 * distances[depot_index][i] for i in range(n_clients)]  # Cost of visiting each client directly
	route_demands = [demands[i] for i in range(n_clients)]

	# Step 2: Compute savings
	savings = []
	for i in range(n_clients):
		for j in range(i + 1, n_clients):
			saving = distances[depot_index][i] + distances[depot_index][j] - distances[i][j]
			savings.append((saving, i, j))
	savings.sort(reverse=True, key=lambda x: x[0])  # Sort savings in descending order

	# Step 3: Merge routes based on savings
	for saving, i, j in savings:
		# Find routes containing i and j
		route_i = next((r for r in routes if i in r), None)
		route_j = next((r for r in routes if j in r), None)

		# Check if merging is possible
		if route_i is not None and route_j is not None and route_i != route_j:
			if sum([demands[i] for i in route_i]) + sum([demands[i] for i in route_j]) <= veh_cap:
				print(routes)
				print(i, j, route_i, route_j)
				# Merge routes
				if route_i[-1] == i and route_j[0] == j:
					merged_route = route_i + route_j
				elif route_i[0] == i and route_j[-1] == j:
					merged_route = route_j + route_i
				elif route_i[-1] == i and route_j[-1] == j:
					merged_route = route_i + route_j.reverse()
				elif route_i[0] == i and route_j[0] == i:
					merged_route = route_j.reverse() + route_i.reverse()
				else:
					continue

				
				# Update route demand and cost

				route_demands[routes.index(route_i)] += route_demands[routes.index(route_j)]
				route_costs[routes.index(route_i)] = (
					route_costs[routes.index(route_i)]
					+ route_costs[routes.index(route_j)]
					- saving
				)
				# Update the routes list
				routes.remove(route_i)
				routes.remove(route_j)
				routes.append(merged_route)

	# Step 4: Convert routes to arcs
	arcs = []
	total_cost = 0
	for route in routes:
		print(route)
		route_arcs = []
		prev = depot_index
		for client in route:
			route_arcs.append((prev, client))
			total_cost += distances[prev][client]
			prev = client
		route_arcs.append((prev, depot_index))
		total_cost += distances[prev][depot_index]
		arcs.append(route_arcs)

	return {
		"arcs": arcs,
		"total_cost": total_cost
	}

def gurobi_vrp_subtour_elim(instance):
	display = False
	sol_gurobi =gurobi_cont_sous_tour(instance, [])
	if(display):
		print("Initial solution")
		plot_solution(instance, sol_gurobi)
	(found, newSousTour) = detect_sous_tour(sol_gurobi, instance)	
	
	print("new sous tours")	
	sousTours=newSousTour
	print(len(sousTours))
	print(sousTours)

	while(found):
		sol_gurobi = gurobi_cont_sous_tour(instance, sousTours)
		(found, newSousTour) = detect_sous_tour(sol_gurobi, instance)
			
		
		sousTours+=newSousTour
		print(str(len(sousTours)) + " sous tours")
		#print(sousTours)
		if(display):
			plot_solution(instance, sol_gurobi)

	return sol_gurobi		

def detect_sous_tour(sol, instance):
	depotindex= instance["num_clients"]
	subtours = []
	for k in range(len(sol["arcs"])):
		arcs = sol["arcs"][k]
		print(arcs)
		graph = {u: v for u, v in arcs}  # Création d'un dictionnaire représentant le cycle
		visited = set()
		

		for start in graph:
			if start not in visited:
				subtour = []
				node = start
				while node not in visited:
					visited.add(node)
					subtour.append(node)
					node = graph[node]

				if node == start:
					if(len(subtour)== len(graph.keys()) or depotindex in subtour):
						print("sous tour ignoree")
						print(subtour)
					else:
						subtours.append(subtour) # Retourne le premier sous-tour détecté
	if(len(subtours)>0):
		print(subtours)
		return (True, subtours)
	else:
		return (False, [])  # Aucun sous-tour détecté (ce qui signifie que la solution est valide)
	

def gurobi_cont_sous_tour(instance, sousTours):
	"""
	Résout le problème de VRP avec Gurobi.

	Returns:
		(dict): Solution avec les installations ouvertes et affectations.
	"""

	print(sousTours)

	num_clients = instance["num_clients"]
	depot = instance["depot"]
	clients = instance["clients"]
	demands = instance["demands"]
	vertices = np.vstack([clients, depot])

	num_veh = instance["n_vehicles"]
	veh_cap = instance["veh_cap"]

	n_clients = len(clients)
	n_vertices = len(vertices)
	idepot = n_vertices - 1

	def dist(point1, point2):
		return math.sqrt((point1[0]- point2[0])*(point1[0]- point2[0]) + (point1[1]- point2[1])*(point1[1]- point2[1]))
	distances = [[dist(i, j) for i in vertices]for j in vertices]
	

	# Modèle Gurobi
	model = gp.Model("VRP")

	# Variables de décision
	x = model.addVars(n_vertices, n_vertices, num_veh, vtype=GRB.BINARY, name="Assign")

	y = model.addVars(n_clients, num_veh, vtype=GRB.BINARY, name="AssignVeh")
	# Fonction objectif : minimiser le coût total
	model.setObjective(
		gp.quicksum(distances[i][j] * x[i, j, k] for i in range(n_vertices) for j in range(n_vertices) for k in range(num_veh)),
		sense=GRB.MINIMIZE
	)

	

	# Contrainte 1 : Capacité des véhicules
	for k in range(num_veh):
		model.addConstr(gp.quicksum(demands[i]*y[i, k] for i in range(n_clients)) <= veh_cap, f"Capacity_{k}")

	# Contrainte 2 : chaque client est desservi par un véhicule
	for i in range(n_clients):
		model.addConstr(gp.quicksum(y[i, k] for k in range(num_veh)) == 1, f"ClientAssigned_{i}")

	# Contrainte 3 : chaque véhicule sort du dépôt
	for k in range(num_veh):
		model.addConstr(gp.quicksum(x[idepot, j, k] for j in range(0, n_vertices)) == 1, f"VehicleLeavesDepot_{k}")
	
	# Contrainte 4 : chaque véhicule entre dans le dépôt
	for k in range(num_veh):
		model.addConstr(gp.quicksum(x[i, idepot, k] for i in range(0, n_vertices)) == 1, f"VehicleReturnsDepot_{k}")

	# Contrainte 5 : chaque véhicule sort d'un client
	for i in range(n_clients):
		for k in range(num_veh):
			model.addConstr(gp.quicksum(x[i, j, k] for j in range(n_vertices)) == y[i, k], f"VehicleLeavesClient_{i}_{k}")
	
	#
	# Contrainte 6 : chaque véhicule entre dans un client
	for i in range(n_clients):
		for k in range(num_veh):
			model.addConstr(gp.quicksum(x[j, i, k] for j in range(n_vertices)) == y[i, k], f"VehicleEntersClient_{i}_{k}")
	
	
	#contraintes d'élimination de sous-tours
	#
	for k in range(num_veh):
		for subtour in sousTours:
			model.addConstr(sum(x[i, j, k] for i in subtour for j in subtour) <= len(subtour) - 1)
	
	
	#Contraintes de loop
	for i in range(n_clients):
		for k in range(num_veh):
			model.addConstr(x[i, i, k] == 0, f"NoLoop_{i}_{k}")
	
	# Résolution
	#model.Params.OutputFlag = 0  # Disable Gurobi output
	model.optimize()

	#imprime la solution
	if model.status == GRB.OPTIMAL:
		print("Solution optimale trouvée")
	else:
		print("Aucune solution optimale trouvée")
	
	#imprime les variables y
	for k in range(num_veh):
		for i in range(n_clients):
			if y[i, k].X > 0.5:
				print(f"Client {i} est desservi par le véhicule {k}")
	#imprime les variables x
	for k in range(num_veh):
		for i in range(n_vertices):
			for j in range(n_vertices):
				if x[i, j, k].X > 0.5:
					print(f"Le véhicule {k} va de {i} à {j}")	
	# Extraction de la solution
	arcs = [ [(i,j)  for i in range(n_vertices) for j in range(n_vertices) if x[i, j, k].X > 0.5] for k in range(num_veh)  ]
	print(arcs)
	total_cost = model.ObjVal
	return {
		"arcs": arcs,
		"total_cost": total_cost
	}

#Imprime la solution
#
def plot_solution(instance,solution):
	arcs = solution["arcs"]
	demands= instance["demands"]
	total_cost = solution["total_cost"]
	depot = instance["depot"]
	num_vehicles = len(arcs)
	"""
	Affiche la solution du problème de localisation d'installations.

	Args:
		instance (dict): L'instance du problème.
		open_facilities (list): Liste des indices des installations ouvertes.
		client_assignments (dict): Dictionnaire {client_id: facility_id}
	"""
	clients = instance["clients"]  # {client_id: np.array([x, y])}
	vertices = np.vstack([clients, depot])
	plt.figure(figsize=(8, 6))

	#affichage des cleints
	for i in range(len(clients)):
		x, y = clients[i]
		plt.scatter(x, y, color='blue', marker='o', label="Clients" if i == 0 else "")
		plt.text(x, y, str(i)+" ("+str(demands[i]), fontsize=9, ha='right', va='bottom')
	
	plt.scatter(depot[0], depot[1], color='red', marker='^', label="Depot")
	plt.text(depot[0], depot[1], "DEP ", fontsize=9, ha='right', va='bottom')
			

	# Affichage des arcs
	colors = plt.cm.get_cmap('tab20', len(arcs))  # Use 'tab20' colormap for more distinct colors
	for k in range(len(arcs)):
		color = colors(k)  # Get a unique color for each vehicle
		for (i, j) in arcs[k]:
			client1 = vertices[i]
			client2 = vertices[j]
			x1, y1 = vertices[i]
			x2, y2 = vertices[j]
			plt.plot([x1, x2], [y1, y2], '-', color=color, alpha=0.7)



 	#Affichage du coût total
	plt.text(
		0.05, 1.05,
		f"Coût total: {total_cost:.2f}\nNombre de véhicules: {num_vehicles}",
		transform=plt.gca().transAxes,
		fontsize=12,
		bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
	)
	# Légende et affichage
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title("Solution du VRP")
	plt.legend()
	plt.grid(True)
	plt.show()

	
def plot_instance(instance):
	"""Affiche une instance du problème de localisation d'installations."""
	plt.figure(figsize=(8, 6))
	plt.scatter(instance["clients"][:, 0], instance["clients"][:, 1], c='blue', marker='o', label="Clients")
	plt.legend()
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title("Instance du problème TSP")
	plt.grid()
	plt.show()


def load_instance_text(prefix="instance"):
	"""
	Charge une instance à partir des fichiers texte :
	- clients.txt
	- facilities.txt
	- transport_costs.txt

	Args:
		prefix (str): Préfixe des fichiers d'entrée.

	Returns:
		dict: L'instance du problème reconstituée.
	"""
	# Lecture des clients
	clients = []
	demands = []
	with open(f"{prefix}/clients.txt", "r") as f:
		for line in f:
			parts = line.strip().split()
			clients.append([float(parts[1]), float(parts[2])])
			demands.append(int(parts[3]))
	clients = np.array(clients)

	depot = clients[0]
	clients = np.delete(clients, 0, axis=0)  # Supprimer le dépôt des clients
	# Lecture des installations
	facilities = []
	fixed_costs = []
	with open(f"{prefix}/facilities.txt", "r") as f:
		for line in f:
			parts = line.strip().split()
			facilities.append([float(parts[1]), float(parts[2])])
			fixed_costs.append(int(parts[3]))
	facilities = np.array(facilities)

	# Lecture de la matrice des coûts
	transport_costs = np.loadtxt(f"{prefix}/transport_costs.txt")

	return {
		"num_facilities": len(facilities),
		"num_clients": len(clients),
		"facilities": facilities,
		"clients": clients,
		"depot": depot,
		"fixed_costs": np.array(fixed_costs),
		"demands": np.array(demands),
		"transport_costs": transport_costs,
		"veh_cap":75,
		"n_vehicles": math.ceil(sum(demands)/75*1.2),
	}



	

if __name__ == "__main__":
	main()