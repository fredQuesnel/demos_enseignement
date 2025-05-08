

import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import json

def save_instance_text(instance, prefix="instance"):
	"""
	Sauvegarde une instance dans trois fichiers texte :
	- clients.txt : ID, position (x, y) et demande.
	- facilities.txt : ID, position (x, y) et coût fixe.
	- transport_costs.txt : Matrice des coûts de transport.

	Args:
		instance (dict): Instance du problème.
		prefix (str): Préfixe des fichiers de sortie.
	"""
	# Sauvegarde des clients
	with open(f"{prefix}clients.txt", "w") as f:
		for i, (pos, demand) in enumerate(zip(instance["clients"], instance["demands"])):
			f.write(f"{i} {pos[0]:.6f} {pos[1]:.6f} {demand}\n")

	# Sauvegarde des installations
	with open(f"{prefix}facilities.txt", "w") as f:
		for i, (pos, cost) in enumerate(zip(instance["facilities"], instance["fixed_costs"])):
			f.write(f"{i} {pos[0]:.6f} {pos[1]:.6f} {cost}\n")

	# Sauvegarde de la matrice des coûts
	np.savetxt(f"{prefix}transport_costs.txt", instance["transport_costs"], fmt="%.6f")

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
	with open(f"{prefix}_clients.txt", "r") as f:
		for line in f:
			parts = line.strip().split()
			clients.append([float(parts[1]), float(parts[2])])
			demands.append(int(parts[3]))
	clients = np.array(clients)

	# Lecture des installations
	facilities = []
	fixed_costs = []
	with open(f"{prefix}_facilities.txt", "r") as f:
		for line in f:
			parts = line.strip().split()
			facilities.append([float(parts[1]), float(parts[2])])
			fixed_costs.append(int(parts[3]))
	facilities = np.array(facilities)

	# Lecture de la matrice des coûts
	transport_costs = np.loadtxt(f"{prefix}_transport_costs.txt")

	return {
		"num_facilities": len(facilities),
		"num_clients": len(clients),
		"facilities": facilities,
		"clients": clients,
		"fixed_costs": np.array(fixed_costs),
		"demands": np.array(demands),
		"transport_costs": transport_costs,
	}

def generate_facility_location_instance(num_facilities=5, num_clients=10, seed=None):
	"""
	Génère une instance aléatoire du problème de localisation d'installations.
	Ajoute des demandes pour les clients.

	Args:
		num_facilities (int): Nombre d'installations potentielles.
		num_clients (int): Nombre de clients.
		seed (int, optional): Graine pour la reproductibilité.

	Returns:
		dict: Un dictionnaire contenant les positions, coûts fixes, demandes et coûts de transport.
	"""
	if seed is not None:
		np.random.seed(seed)

	# Génération des positions
	facilities = np.random.rand(num_facilities, 2) * 100
	clients = np.random.rand(num_clients, 2) * 100

	# Coûts fixes aléatoires pour ouvrir chaque installation
	fixed_costs = np.random.randint(50, 200, size=num_facilities)

	# Demandes aléatoires des clients
	demands = np.random.randint(10, 50, size=num_clients)

	# Matrice des coûts de transport (distance euclidienne)
	transport_costs = np.linalg.norm(
		facilities[:, np.newaxis, :] - clients[np.newaxis, :, :], axis=2
	)

	instance = {
		"num_facilities": num_facilities,
		"num_clients": num_clients,
		"facilities": facilities,
		"clients": clients,
		"fixed_costs": fixed_costs,
		"demands": demands,
		"transport_costs": transport_costs,
	}

	return instance

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

def main():
	"""Point d'entrée du script."""
	sizes = [[10, 20], [10, 100], [20, 1000], [20, 200]]
	for N, M in sizes:
		folder = "N"+str(N)+"_C"+str(M)+"/"
		os.makedirs(folder)

		instance = generate_facility_location_instance(num_facilities=N, num_clients=M, seed=42)
		#plot_instance(instance)
		save_instance_text(instance, folder)


if __name__ == "__main__":
	main()
