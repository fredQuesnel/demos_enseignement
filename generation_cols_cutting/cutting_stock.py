#Credits : Code provided by github user demirayonur
# https://github.com/demirayonur/Column-Generation/blob/main/ColumnGeneration_CuttingStockProblem.ipynb
#

from gurobipy import GRB
import gurobipy as gp
import numpy as np
np.random.seed(42)
import math


#create an instance of the knapsack problem
#
class Instance(object):
		
		def __init__(self, num_order =3, min_order_len=10, max_order_len=64,
								min_demand=1, max_demand=500, roll_length=215):
				
				# generates an instance which guarantees feasibility.
				self.n = num_order
				if(num_order==3):
					self.order_lens = [64, 60, 35]
				else:	
					self.order_lens = np.random.randint(min_order_len, max_order_len+1, size=self.n)
				self.demands = np.random.randint(min_demand, max_demand+1, size=self.n)
				self.m = np.sum(self.demands)
				self.roll_len = roll_length
		
		def summarize(self):
				print("Problem instance with ", self.n, " orders and ", self.m, "rolls")
				print("-"*47)
				print("\nOrders:\n")
				for i, order_len in enumerate(self.order_lens):
						print("\tOrder ", i, ": length= ", order_len, " demand=", self.demands[i])
				print("\nRoll Length: ", self.roll_len)


def main(): 
	instance = Instance(15)
	#instance = Instance()
	instance.summarize()			

	optimize_kantorovich(instance)
	#optimize_all_patterns(instance)

	#history = column_generation_frac(instance)
	#history = column_generation(instance)

	
	
	#history = column_generation_frac(instance)
	#history = column_generation(instance)

	#patterns=all_patterns(instance)
	#print(patterns)

# create a set of initial patterns
# to solve the first restricted master problem
#
def generate_initial_patterns(ins_:Instance):
		patterns = []
		#each initial pattern serves a single order as much as possible
		for i in range(ins_.n):
				pattern_ = list(np.zeros(ins_.n).astype(int))
				pattern_[i] = int(ins_.roll_len/ins_.order_lens[i])
				patterns.append(pattern_)
		return patterns
		
# defines the restricted master problem (RMP)
# This is somewhat inefficient : ideally, we wouldn't recreate the RMP each iteration
# but rather modify it by adding the necessary variables. 
#		
def define_master_problem(ins_:Instance, patterns):
		
		
		n_pattern = len(patterns) 
		pattern_range = range(n_pattern) 
		order_range = range(ins_.n)
		patterns = np.array(patterns, dtype=int)
		master_problem = gp.Model("master problem")
		master_problem.setParam("OutputFlag",0)
		
		# decision variables
		# variables are continuous in the RMP
		lambda_ = master_problem.addVars(pattern_range,
																		 vtype=GRB.CONTINUOUS, 
																		 obj=np.ones(n_pattern),
																		 name="lambda")
		
		# direction of optimization (min or max)
		master_problem.modelSense = GRB.MINIMIZE
		
		# demand satisfaction constraint
		for i in order_range:
				master_problem.addConstr(sum(patterns[p,i]*lambda_[p] for p in pattern_range) >= ins_.demands[i],
																 "Demand[%d]" %i)
						
		# solve
		return master_problem

#Defines an integer version of the RMP
# This is used in the final iteration of the "restricted master heuristic"
def define_IMP(ins_:Instance, patterns):
		
		n_pattern = len(patterns)
		pattern_range = range(n_pattern)
		order_range = range(ins_.n)
		patterns = np.array(patterns, dtype=int)
		master_problem = gp.Model("master problem")
		#master_problem.setParam("OutputFlag",0)
		
		# decision variables
		lambda_ = master_problem.addVars(pattern_range,
																		 vtype=GRB.INTEGER,
																		 obj=np.ones(n_pattern),
																		 name="lambda")
		
		# direction of optimization (min or max)
		master_problem.modelSense = GRB.MINIMIZE
		
		# demand satisfaction constraint
		for i in order_range:
				master_problem.addConstr(sum(patterns[p,i]*lambda_[p] for p in pattern_range) >= ins_.demands[i],
																 "Demand[%d]" %i)
						
		# solve
		return master_problem
		
		
def define_subproblem(ins_:Instance, duals):
		
		order_range = range(ins_.n)
		subproblem = gp.Model("subproblem")
		subproblem.setParam("OutputFlag",0)
		
		# decision variables
		x = subproblem.addVars(order_range,
													 vtype=GRB.INTEGER,
													 obj=duals,
													 name="x")
		
		# direction of optimization (min or max)
		subproblem.modelSense = GRB.MAXIMIZE
				
		# Length constraint
		subproblem.addConstr(sum(ins_.order_lens[i] * x[i] for i in order_range) <= ins_.roll_len)
		
		return subproblem		
		
		
def print_solution(master, patterns):
		use = [i.x for i in master.getVars()]
		for i, p in enumerate(patterns):
				if use[i]>0:
						print('Pattern ', i, ': how often we should cut: ', use[i])
						print('----------------------')
						for j,order in enumerate(p):
								if order >0:
										print('order ', j, ' how much: ', order)
						print()		
						
						
def column_generation(ins_:Instance):
		
		patterns = generate_initial_patterns(ins_)
		objVal_history = []
		iterNb=0
		while True:
				print("Iteration "+str(iterNb))
				
				master_problem = define_master_problem(ins_, patterns)
				master_problem.optimize()
				objVal_history.append(master_problem.objVal)
				dual_variables = np.array([constraint.pi for constraint in master_problem.getConstrs()])
				subproblem = define_subproblem(ins_, dual_variables)
				subproblem.optimize()
				if subproblem.objVal < 1 + 1e-6:
						break
				patterns.append([i.x for i in subproblem.getVars()])
				pattern = [i.x for i in subproblem.getVars()]
				print(pattern)
				iterNb+=1		
				
		lpSol = np.array([i.x for i in master_problem.getVars()]).sum()
		#solve integer master problem
		integer_master_problem = define_IMP(ins_, patterns)
		integer_master_problem.optimize()
		objVal_history.append(integer_master_problem.objVal)
				
		print_solution(integer_master_problem, patterns)
		print('Total number of patterns generated : '+str(len(patterns))) 
		print("lower bound "+ str(lpSol))
		print('Total number of rolls used: ', np.array([i.x for i in integer_master_problem.getVars()]).sum())
		return objVal_history			
 
def column_generation_frac(ins_:Instance):
		
		patterns = generate_initial_patterns(ins_)
		objVal_history = []
		while True:
				master_problem = define_master_problem(ins_, patterns)
				master_problem.optimize()
				objVal_history.append(master_problem.objVal)
				dual_variables = np.array([constraint.pi for constraint in master_problem.getConstrs()])
				subproblem = define_subproblem(ins_, dual_variables)
				subproblem.optimize()
				if subproblem.objVal < 1 + 1e-6:
						break
				patterns.append([i.x for i in subproblem.getVars()])
						
					 
		print_solution(master_problem, patterns)
		print('Total number of rolls used: ', np.array([i.x for i in master_problem.getVars()]).sum())
		return objVal_history			

def optimize_all_patterns(ins_):
	patterns = all_patterns(ins_)
	#solve integer master problem
	print("defining IMP")
	integer_master_problem = define_IMP(ins_, patterns)
	print("optimizing")
	integer_master_problem.optimize()
			
	print_solution(integer_master_problem, patterns)
	print('Total number of rolls used: ', np.array([i.x for i in integer_master_problem.getVars()]).sum())

def optimize_kantorovich(ins_:Instance):
		
		# model
		model = gp.Model("kantorovich_formulation")
		
		# sets (ranges)
		rolls = range(ins_.m)
		orders = range(ins_.n)
		
		# decision variables
		use_roll = model.addVars(rolls,
														 vtype=GRB.BINARY,
														 obj=np.ones(ins_.m),
														 name="X")
		
		how_much_use = model.addVars(orders, rolls, 
																 vtype=GRB.INTEGER,
																 obj= np.zeros((ins_.n, ins_.m)), 
																 name="Y")
		
		'''
		We could also use looping to 
		generate decision variables: 
		
		Example for "use_roll":
		-----------------------
		use_roll = []
		for j in rolls:
				use_roll.append(model.addVar(vtype=GRB.BINARY,
																		 obj=1,
																		 name="X[%d]" %p ))
		Example for "how_much_use":
		---------------------------
		how_much_use = []
		for j in rolls:
				how_much_use.append([])
				for i in orders:
						how_much_use[j].append(model.addVar(...))
		'''
		
		# direction of optimization (min or max)
		model.modelSense = GRB.MINIMIZE
		
		# demand satisfaction constraint
		model.addConstrs(
										(how_much_use.sum(i, '*') == ins_.demands[i] for i in orders), 
										"Demand"
										)
		
		# length constraint of a roll
		for j in rolls:
				model.addConstr(sum(how_much_use[i,j]*ins_.order_lens[i] for i in orders)
												<= ins_.roll_len,
												"Length[%d]" %j)
		
		# x-y link
		for i in orders:
				for j in rolls:
						model.addConstr(how_much_use[i,j] <= use_roll[j]*ins_.demands[i])
		
		if False:
				model.write('kantorovich.lp') # in case you want to write the model
		
		# solve
		model.optimize()
		
		# display objective function value
		print('\nTotal Number of Rolls Used: ', model.objVal)
		


		

def all_patterns(ins_:Instance):

	
		patterns = []
		pattern = []
		patterns = subpatterns(ins_, pattern, 0, ins_.roll_len)
		
		if(len(patterns)<=100):
			print(patterns)
		print("il y a "+str(len(patterns))+" patterns")
		return patterns
		
def subpatterns(ins_:Instance, mainPattern, index, remainingLen):
	
	patterns = []
	
	maxi = int(remainingLen/ins_.order_lens[index])
	
	#last index : fill all possible
	if(index == ins_.n -1):
		pattern = mainPattern.copy()
		pattern.append(maxi)
		patterns.append(pattern)
	else:	
		for i in range(maxi+1):
			pattern = mainPattern.copy()
			pattern.append(i)
			patterns = patterns+subpatterns(ins_, pattern, index+1, remainingLen-i*ins_.order_lens[index])
	
	return patterns		


if __name__ == "__main__":

		main()



#print(history)		