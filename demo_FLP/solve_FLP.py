import sys
import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def main():
	folder = sys.argv[1]
	instance = load_instance_text(folder)
	
	#plot_instance(instance)

	#------------------------------
	#version sans capacité
	#------------------------------

	#resout avec gurobi, imprime la solution
	#sol_gurobi = gurobi_facility_location(instance)
	#plot_solution(instance, sol_gurobi)
	
	#exit(0)
	#resout avec heuristique, imprime la solution
	
	#sol_greedy = greedy_facility_location(instance)
	#plot_solution(instance, sol_greedy)

	#------------------------------
	#version avec capacité
	#------------------------------
	sol_cap = heuristic_capacity_facility_location(instance)
	#plot_solution(instance, sol_cap)
	sol_cap = OneOptFLPCap(instance, sol_cap)
	plot_solution(instance, sol_cap)

	#resout avec gurobi, imprime la solution
	#sol_gurobi_cap = gurobi_facility_location_cap(instance)
	#plot_solution(instance, sol_gurobi_cap)

	


def OneOptFLPCap(instance, solution):
	capacity = 5000
	num_facilities = instance["num_facilities"]
	num_clients = instance["num_clients"]
	facilities = instance["facilities"]
	clients = instance["clients"]
	facility_costs = instance["fixed_costs"]
	transport_costs = instance["transport_costs"]
	n_clients = len(clients)
	n_facilities = len(facilities)
	client_demands = instance["demands"]

	open_facilities = solution["open_facilities"]
	client_assignments = solution["client_assignments"]
	total_cost = solution["total_cost"]

	print(open_facilities)
	capacities = [capacity for i in range(len(facilities))]

	for i in range(len(clients)):
		assign = client_assignments[i]
		capacities[assign] -= client_demands[i]
	
	while(True):
		found = False
		for i in range(len(clients)):
			
			for j in open_facilities:
				if(transport_costs[j][i]< transport_costs[client_assignments[i], i] and capacities[j]< client_demands[i]):
					total_cost+=transport_costs[j][i]- transport_costs[client_assignments[i], i]
					capacities[j]-= client_demands[i]
					capacities[client_assignments[i]] += client_demands[i]
					client_assignments[i]=j
					print(total_cost)
					found = True
		if(not found):
			break	

	return {
		"open_facilities": open_facilities,
		"client_assignments": client_assignments,
		"total_cost": total_cost
	}



def heuristic_capacity_facility_location(instance):
	

	capacity = 5000
	num_facilities = instance["num_facilities"]
	num_clients = instance["num_clients"]
	facilities = instance["facilities"]
	clients = instance["clients"]
	facility_costs = instance["fixed_costs"]
	transport_costs = instance["transport_costs"]
	n_clients = len(clients)
	n_facilities = len(facilities)
	client_demands = instance["demands"]
	"""
	Heuristique gloutonne pour le Facility Location Problem avec capacités.

	Args:
		facilities (np.array): Coordonnées des installations (n_facilities, 2).
		facility_costs (np.array): Coûts fixes des installations (n_facilities,).
		facility_capacities (np.array): Capacités des installations (n_facilities,).
		clients (np.array): Coordonnées des clients (n_clients, 2).
		client_demands (np.array): Demande de chaque client (n_clients,).
		transport_costs (np.array): Matrice des coûts de transport (n_clients, n_facilities).

	Returns:
		(dict): Solution avec les installations ouvertes, affectations et coût total.
	"""

	n_clients = len(clients)
	n_facilities = len(facilities)

	# Trier les installations par coût fixe croissant
	facility_order = np.argsort(facility_costs)
	print(facility_order)

	# Trier les clients par demande décroissante
	client_order = np.argsort(-client_demands)

	# Suivi des installations ouvertes et de leur capacité restante
	open_facilities = []
	remaining_capacity = np.zeros(n_facilities)
	client_assignments = {}

	# Attribuer les clients aux installations
	for i in client_order:  # Parcourir les clients du plus gros au plus petit
		assigned = False
		bestFacility=None
		bestCost = float("inf") 
		print("\n\nclient "+str(i))
		print(client_demands[i])
		for j in facility_order:  # Parcourir les installations du moins cher au plus cher
			#print("facility "+str(j))
			#print("rem : "+str(remaining_capacity[j]))
			#print("open? "+str(j in open_facilities))

			if j in open_facilities and remaining_capacity[j] >= client_demands[i]:
				#print("open facility "+str(j))
				
				cost = transport_costs[j, i]
				#print("  cost : "+str(cost))
				if(cost < bestCost):
					#print("ici")
					bestFacility = j
					bestCost = cost
				
			elif j not in open_facilities and capacity >= client_demands[i]:
				#print("closed facility "+str(j))
				# Ouvrir une nouvelle installation si nécessaire
				cost = facility_costs[j]+transport_costs[j, i]
				#print("  cost : "+str(cost))
				if(cost < bestCost):
					bestFacility = j
					bestCost = cost
		print("best : ")					
		print(bestFacility)
		if(bestFacility is not None):
			if( bestFacility not in open_facilities):
				open_facilities.append(bestFacility)
				remaining_capacity[bestFacility] = capacity
			remaining_capacity[bestFacility] = remaining_capacity[bestFacility]  - client_demands[i]

			client_assignments[i] = bestFacility
			print(remaining_capacity)
			assigned = True


		# Si aucun site ne peut accueillir ce client, on l'ignore (erreur potentielle si le problème est mal posé)
		if not assigned:
			raise ValueError(f"Impossible d'assigner le client {i} à une installation avec la capacité disponible.")

	# Calcul du coût total
	total_cost = sum(facility_costs[j] for j in open_facilities) + sum(transport_costs[client_assignments[i], i] for i in range(n_clients))

	return {
		"open_facilities": open_facilities,
		"client_assignments": client_assignments,
		"total_cost": total_cost
	}




def gurobi_facility_location(instance):
	"""
	Résout le problème de facility location avec Gurobi.

	Args:
		facilities (np.array): Coordonnées des installations (n_facilities, 2).
		facility_costs (np.array): Coûts fixes des installations (n_facilities,).
		clients (np.array): Coordonnées des clients (n_clients, 2).
		client_demands (np.array): Demande de chaque client (n_clients,).
		transport_costs (np.array): Matrice des coûts de transport (n_clients, n_facilities).

	Returns:
		(dict): Solution avec les installations ouvertes et affectations.
	"""

	num_facilities = instance["num_facilities"]
	num_clients = instance["num_clients"]
	facilities = instance["facilities"]
	clients = instance["clients"]
	facility_costs = instance["fixed_costs"]
	transport_costs = instance["transport_costs"]
	n_clients = len(clients)
	n_facilities = len(facilities)

	# Modèle Gurobi
	model = gp.Model("FacilityLocation")

	# Variables de décision
	y = model.addVars(n_facilities, vtype=GRB.BINARY, name="OpenFacility")
	x = model.addVars(n_clients, n_facilities, vtype=GRB.BINARY, name="Assign")

	# Fonction objectif : minimiser le coût total
	model.setObjective(
		gp.quicksum(facility_costs[j] * y[j] for j in range(n_facilities)) +
		gp.quicksum(transport_costs[j, i] * x[i, j] for i in range(n_clients) for j in range(n_facilities)),
		sense=GRB.MINIMIZE
	)

	# Contrainte 1 : Chaque client est assigné à exactement une installation
	for i in range(n_clients):
		model.addConstr(gp.quicksum(x[i, j] for j in range(n_facilities)) == 1, f"AssignClient_{i}")

	# Contrainte 2 : Un client ne peut être assigné qu'à une installation ouverte
	for i in range(n_clients):
		for j in range(n_facilities):
			model.addConstr(x[i, j] <= y[j], f"OpenFacilityIfAssigned_{i}_{j}")

	# Résolution
	model.optimize()

	# Extraction de la solution
	open_facilities = [j for j in range(n_facilities) if y[j].X > 0.5]
	client_assignments = [ next(j for j in range(n_facilities) if x[i, j].X > 0.5) for i in range(n_clients)]
	total_cost = model.ObjVal

	return {
		"open_facilities": open_facilities,
		"client_assignments": client_assignments,
		"total_cost": total_cost
	}


def gurobi_facility_location_cap(instance):
	"""
	Résout le problème de facility location avec Gurobi.

	Args:
		facilities (np.array): Coordonnées des installations (n_facilities, 2).
		facility_costs (np.array): Coûts fixes des installations (n_facilities,).
		clients (np.array): Coordonnées des clients (n_clients, 2).
		client_demands (np.array): Demande de chaque client (n_clients,).
		transport_costs (np.array): Matrice des coûts de transport (n_clients, n_facilities).

	Returns:
		(dict): Solution avec les installations ouvertes et affectations.
	"""
	capacity=500

	num_facilities = instance["num_facilities"]
	num_clients = instance["num_clients"]
	facilities = instance["facilities"]
	clients = instance["clients"]
	demands = instance["demands"]
	facility_costs = instance["fixed_costs"]
	transport_costs = instance["transport_costs"]
	n_clients = len(clients)
	n_facilities = len(facilities)

	# Modèle Gurobi
	model = gp.Model("FacilityLocation")

	# Variables de décision
	y = model.addVars(n_facilities, vtype=GRB.BINARY, name="OpenFacility")
	x = model.addVars(n_clients, n_facilities, vtype=GRB.BINARY, name="Assign")

	# Fonction objectif : minimiser le coût total
	model.setObjective(
		gp.quicksum(facility_costs[j] * y[j] for j in range(n_facilities)) +
		gp.quicksum(transport_costs[j, i] * x[i, j] for i in range(n_clients) for j in range(n_facilities)),
		sense=GRB.MINIMIZE
	)

	# Contrainte 1 : Chaque client est assigné à exactement une installation
	for i in range(n_clients):
		model.addConstr(gp.quicksum(x[i, j] for j in range(n_facilities)) == 1, f"AssignClient_{i}")

	# Contrainte 2 : Un client ne peut être assigné qu'à une installation ouverte
	for j in range(n_facilities):
		model.addConstr(gp.quicksum(demands[i]*x[i, j] for i in range(n_clients)) <= capacity*y[j], f"OpenFacilityIfAssigned_{i}_{j}")

	# Résolution
	model.optimize()

	# Extraction de la solution
	open_facilities = [j for j in range(n_facilities) if y[j].X > 0.5]
	client_assignments = [ next(j for j in range(n_facilities) if x[i, j].X > 0.5) for i in range(n_clients)]
	total_cost = model.ObjVal

	return {
		"open_facilities": open_facilities,
		"client_assignments": client_assignments,
		"total_cost": total_cost
	}

#Imprime la solution
#
def plot_solution(instance,solution):
	open_facilities = solution["open_facilities"]
	client_assignments = solution["client_assignments"]
	total_cost = solution["total_cost"]
	
		
	"""
	Affiche la solution du problème de localisation d'installations.

	Args:
		instance (dict): L'instance du problème.
		open_facilities (list): Liste des indices des installations ouvertes.
		client_assignments (dict): Dictionnaire {client_id: facility_id}
	"""
	clients = instance["clients"]  # {client_id: np.array([x, y])}
	facilities = instance["facilities"]  # {facility_id: np.array([x, y])}

	plt.figure(figsize=(8, 6))

	# Affichage des clients et liens vers les installations
	for i in range(len(clients)):
		x, y = clients[i]
		plt.scatter(x, y, color='blue', marker='o', label="Clients" if i == 0 else "")
		facility_id = client_assignments[i]
		facility_pos = facilities[facility_id]
		plt.plot([x, facility_pos[0]], [y, facility_pos[1]], 'k--', alpha=0.5)  # Lien client-installation

	# Affichage des installations (ouvertes et fermées)
	for i in range(len(facilities)):
		x, y = facilities[i]
		if i in open_facilities:
			plt.scatter(x, y, color='red', marker='s', s=100, label="Installations ouvertes" if i == open_facilities[0] else "")
		else:
			plt.scatter(x, y, color='gray', marker='s', s=100, alpha=0.5, label="Installations fermées" if i == 0 else "")

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
	plt.scatter(instance["facilities"][:, 0], instance["facilities"][:, 1], c='red', marker='s', label="Installations")
	plt.scatter(instance["clients"][:, 0], instance["clients"][:, 1], c='blue', marker='o', label="Clients")
	plt.legend()
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title("Instance du problème de localisation d'installations")
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