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
	
	plot_instance(instance)

	#------------------------------
	#version sans capacité
	#------------------------------

	#resout avec gurobi, imprime la solution
	#sol_gurobi = gurobi_tsp_relax(instance)
	#plot_solution(instance, sol_gurobi)
	
	#resout avec gurobi, imprime la solution
	start_time = time.time()
	sol_gurobi = gurobi_tsp_miller(instance)
	end_time = time.time()
	plot_solution(instance, sol_gurobi)
	print(f"Execution time: {end_time - start_time:.2f} seconds")
	#start_time = time.time()
	
	#sol_gurobi = gurobi_tsp_subtour_elim(instance)
	#end_time = time.time()
	#plot_solution(instance, sol_gurobi)
	
	
	print(f"Execution time: {end_time - start_time:.2f} seconds")
	#exit(0)
	#resout avec heuristique, imprime la solution
	
	#sol_greedy = greedy_facility_location(instance)
	#plot_solution(instance, sol_greedy)

	#resout avec gurobi, imprime la solution
	#sol_gurobi_cap = gurobi_facility_location_cap(instance)
	#plot_solution(instance, sol_gurobi_cap)

def gurobi_tsp_subtour_elim(instance):
	display = False
	sol_gurobi = gurobi_tsp_relax(instance)

	(found, newSousTour) = detect_sous_tour(sol_gurobi)	
	if(display):
		plot_solution(instance, sol_gurobi)
	print("new sous tours")	
	
	sousTours=newSousTour
	print(len(sousTours))
	print(sousTours)

	while(found):
		sol_gurobi = gurobi_cont_sous_tour(instance, sousTours)
		(found, newSousTour) = detect_sous_tour(sol_gurobi)
			
		
		sousTours+=newSousTour
		print(str(len(sousTours)) + " sous tours")
		#print(sousTours)
		if(display):
			plot_solution(instance, sol_gurobi)

	return sol_gurobi		

def detect_sous_tour(sol):
	arcs = sol["arcs"]
	graph = {u: v for u, v in arcs}  # Création d'un dictionnaire représentant le cycle
	visited = set()
	subtours = []

	for start in graph:
		if start not in visited:
			subtour = []
			node = start
			while node not in visited:
				visited.add(node)
				subtour.append(node)
				node = graph[node]

			if node == start:
				if(len(subtour)== len(graph.keys())):
					return (False, [])
				else:
					subtours.append(subtour) # Retourne le premier sous-tour détecté
	if(len(subtours)>0):
		return (True, subtours)
	else:
		return (False, [])  # Aucun sous-tour détecté (ce qui signifie que la solution est valide)
		
def gurobi_tsp_miller(instance):
	"""
	Résout le problème de TSP location avec Gurobi.

	Args:
		facilities (np.array): ignore
		facility_costs (np.array): ignore
		clients (np.array): Coordonnées des clients (n_clients, 2).
		client_demands (np.array): Demande de chaque client (n_clients,).
		transport_costs (np.array): Matrice des coûts de transport (n_clients, n_facilities).

	Returns:
		(dict): Solution avec les installations ouvertes et affectations.
	"""

	num_clients = instance["num_clients"]
	clients = instance["clients"]
	n_clients = len(clients)
	#n_facilities = len(facilities)

	def dist(point1, point2):
		return math.sqrt((point1[0]- point2[0])*(point1[0]- point2[0]) + (point1[1]- point2[1])*(point1[1]- point2[1]))
	distances = [[dist(i, j) for i in clients]for j in clients]
	print(distances)
	
	# Modèle Gurobi
	model = gp.Model("FacilityLocation")

	# Variables de décision
	x = model.addVars(n_clients, n_clients, vtype=GRB.BINARY, name="Assign")
	u = model.addVars(n_clients, vtype=GRB.CONTINUOUS, name = "miller")
	# Fonction objectif : minimiser le coût total
	model.setObjective(
		gp.quicksum(distances[j][i] * x[i, j] for i in range(n_clients) for j in range(n_clients)),
		sense=GRB.MINIMIZE
	)

	# Contrainte 1 : on sort de chaque client
	for i in range(n_clients):
		model.addConstr(gp.quicksum(x[i, j] for j in range(n_clients)) == 1, f"AssignClient_{i}")

	# Contrainte 2 : on entre dans chaque client
	for i in range(n_clients):
		model.addConstr(gp.quicksum(x[j,i] for j in range(n_clients)) == 1, f"AssignClientB_{i}")

	#contrainte 3 x_ii = 0
	for i in range(n_clients):
		model.addConstr(x[i, i]==0, f"NoLoop_{i}")

	#contrainte 4 : de miller
	for i in range(n_clients):
		for j in range(n_clients):
			if(j != 0 and i !=j):
				model.addConstr(u[i]- u[j]+n_clients*x[i,j] <= n_clients-1)

	# Résolution
	model.optimize()

	# Extraction de la solution
	arcs = [ (i,j)  for i in range(n_clients) for j in range(n_clients) if x[i, j].X > 0.5 ]
	print(arcs)
	total_cost = model.ObjVal
	return {
		"arcs": arcs,
		"total_cost": total_cost
	}





def gurobi_cont_sous_tour(instance, sousTours):
	"""
	Résout le problème de TSP location avec Gurobi.

	Returns:
		(dict): Solution avec les installations ouvertes et affectations.
	"""

	num_clients = instance["num_clients"]
	clients = instance["clients"]
	n_clients = len(clients)
	#n_facilities = len(facilities)

	def dist(point1, point2):
		return math.sqrt((point1[0]- point2[0])*(point1[0]- point2[0]) + (point1[1]- point2[1])*(point1[1]- point2[1]))
	distances = [[dist(i, j) for i in clients]for j in clients]
	
	# Modèle Gurobi
	model = gp.Model("FacilityLocation")

	# Variables de décision
	x = model.addVars(n_clients, n_clients, vtype=GRB.BINARY, name="Assign")

	# Fonction objectif : minimiser le coût total
	model.setObjective(
		gp.quicksum(distances[j][i] * x[i, j] for i in range(n_clients) for j in range(n_clients)),
		sense=GRB.MINIMIZE
	)

	# Contrainte 1 : on sort de chaque client
	for i in range(n_clients):
		model.addConstr(gp.quicksum(x[i, j] for j in range(n_clients)) == 1, f"AssignClient_{i}")

	# Contrainte 2 : on entre dans chaque client
	for i in range(n_clients):
		model.addConstr(gp.quicksum(x[j,i] for j in range(n_clients)) == 1, f"AssignClient_{i}")

	#x_ii = 0
	for i in range(n_clients):
		model.addConstr(x[i, i]==0, f"AssignClient_{i}")

	#contraintes d'élimination de sous-tours
	#
	for subtour in sousTours:
		model.addConstr(sum(x[i, j] for i in subtour for j in subtour if i != j) <= len(subtour) - 1)
	# Résolution
	model.Params.OutputFlag = 0  # Disable Gurobi output
	model.optimize()

	# Extraction de la solution
	arcs = [ (i,j)  for i in range(n_clients) for j in range(n_clients) if x[i, j].X > 0.5 ]
	total_cost = model.ObjVal
	return {
		"arcs": arcs,
		"total_cost": total_cost
	}


def gurobi_tsp_relax(instance):
	"""
	Résout le problème de TSP location avec Gurobi.

	Args:
		facilities (np.array): ignore
		facility_costs (np.array): ignore
		clients (np.array): Coordonnées des clients (n_clients, 2).
		client_demands (np.array): Demande de chaque client (n_clients,).
		transport_costs (np.array): Matrice des coûts de transport (n_clients, n_facilities).

	Returns:
		(dict): Solution avec les installations ouvertes et affectations.
	"""

	num_clients = instance["num_clients"]
	clients = instance["clients"]
	n_clients = len(clients)
	#n_facilities = len(facilities)

	def dist(point1, point2):
		return math.sqrt((point1[0]- point2[0])*(point1[0]- point2[0]) + (point1[1]- point2[1])*(point1[1]- point2[1]))
	distances = [[dist(i, j) for i in clients]for j in clients]
	
	# Modèle Gurobi
	model = gp.Model("FacilityLocation")

	# Variables de décision
	x = model.addVars(n_clients, n_clients, vtype=GRB.BINARY, name="Assign")

	# Fonction objectif : minimiser le coût total
	model.setObjective(
		gp.quicksum(distances[j][i] * x[i, j] for i in range(n_clients) for j in range(n_clients)),
		sense=GRB.MINIMIZE
	)

	# Contrainte 1 : on sort de chaque client
	for i in range(n_clients):
		model.addConstr(gp.quicksum(x[i, j] for j in range(n_clients)) == 1, f"AssignClient_{i}")

	# Contrainte 2 : on entre dans chaque client
	for i in range(n_clients):
		model.addConstr(gp.quicksum(x[j,i] for j in range(n_clients)) == 1, f"AssignClient_{i}")

	#x_ii = 0
	for i in range(n_clients):
		model.addConstr(x[i, i]==0, f"AssignClient_{i}")

	# Résolution
	model.optimize()

	# Extraction de la solution
	arcs = [ (i,j)  for i in range(n_clients) for j in range(n_clients) if x[i, j].X > 0.5 ]
	total_cost = model.ObjVal
	return {
		"arcs": arcs,
		"total_cost": total_cost
	}


#Imprime la solution
#
def plot_solution(instance,solution):
	arcs = solution["arcs"]
	total_cost = solution["total_cost"]
	
		
	"""
	Affiche la solution du problème de localisation d'installations.

	Args:
		instance (dict): L'instance du problème.
		open_facilities (list): Liste des indices des installations ouvertes.
		client_assignments (dict): Dictionnaire {client_id: facility_id}
	"""
	clients = instance["clients"]  # {client_id: np.array([x, y])}

	plt.figure(figsize=(8, 6))

	#affichage des cleints
	for i in range(len(clients)):
		x, y = clients[i]
		plt.scatter(x, y, color='blue', marker='o', label="Clients" if i == 0 else "")
		plt.text(x, y, str(i))
		

	# Affichage des arcs
	for (i,j) in arcs:
		client1 = clients[i]
		client2 = clients[j]
		x1, y1 = clients[i]
		x2, y2 = clients[j]
		plt.plot([x1, x2], [y1, y2], 'k--', alpha=0.5) 

 	#Affichage du coût total
	plt.text(0.05, 0.95, f"Coût total: {total_cost:.2f}", transform=plt.gca().transAxes,
			 fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
	# Légende et affichage
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title("Solution du Facility Location Problem")
	plt.legend()
	plt.grid(True)
	plt.show()

def greedy_facility_location(instance):
	printIntermediate = True
	"""
	Algorithme glouton pour le problème de localisation d'installations.
	À chaque itération, on ouvre l'installation qui minimise le coût total.
	
	Args:
		instance (dict): Instance du problème.

	Returns:
		list: Liste des installations ouvertes.
	"""
	num_facilities = instance["num_facilities"]
	num_clients = instance["num_clients"]
	facilities = instance["facilities"]
	clients = instance["clients"]
	fixed_costs = instance["fixed_costs"]
	transport_costs = instance["transport_costs"]
	# Initialisation
	open_facilities = []
	assigned_facility = np.full(num_clients, -1)  # -1 = non attribué
	total_cost = float("inf")  # Coût initial très grand

	while True:

		#imprime les solutions intermediaires
		#
		
		best_facility = None
		best_new_cost = total_cost

		# Tester chaque installation fermée
		for facility in range(num_facilities):
			
			if facility in open_facilities:
				continue  # Déjà ouverte

			new_total_cost = compute_total_cost(instance, open_facilities+[facility])


			# Vérifier si c'est la meilleure option trouvée
			if new_total_cost < best_new_cost:
				best_new_cost = new_total_cost
				best_facility = facility


		# Si aucune amélioration, arrêter
		if best_facility is None:
			break

		# Ouvrir la meilleure installation trouvée
		open_facilities.append(best_facility)
		total_cost = best_new_cost

		# Mettre à jour l'affectation finale des clients
		for client in range(num_clients):
			if transport_costs[best_facility, client] < (transport_costs[assigned_facility[client], client] if assigned_facility[client] != -1 else float("inf")):
				assigned_facility[client] = best_facility
		
		if(printIntermediate):
			partialSol = {"open_facilities": open_facilities,
			"client_assignments": assigned_facility,
			"total_cost": total_cost
			}
			plot_solution(instance, partialSol)



	return {
		"open_facilities": open_facilities,
		"client_assignments": assigned_facility,
		"total_cost": total_cost
	}

def compute_total_cost(instance, open_facilities):
	"""
	Calcule le coût total en fonction des installations ouvertes.

	Args:
		instance (dict): L'instance du problème.
		open_facilities (list): Liste des indices des installations ouvertes.

	Returns:
		float: Le coût total (fixe + transport).
	"""
	num_clients = instance["num_clients"]
	transport_costs = instance["transport_costs"]
	fixed_costs = instance["fixed_costs"]

	# Si aucune installation n'est ouverte, coût infini
	if not open_facilities:
		return float("inf")

	# Coût fixe : somme des coûts d'ouverture des installations ouvertes
	fixed_cost = sum(fixed_costs[f] for f in open_facilities)

	# Attribution des clients aux installations ouvertes (choisir la plus proche)
	total_transport_cost = 0
	for client in range(num_clients):
		# Trouver la meilleure installation pour ce client
		best_cost = min(transport_costs[f, client] for f in open_facilities)
		total_transport_cost += best_cost

	# Coût total = coût fixe + coût de transport
	return fixed_cost + total_transport_cost



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
		"fixed_costs": np.array(fixed_costs),
		"demands": np.array(demands),
		"transport_costs": transport_costs,
	}



	

if __name__ == "__main__":
	main()