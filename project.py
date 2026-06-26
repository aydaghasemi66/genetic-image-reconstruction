from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Image preparation
# ----------------------------------------------------------------------
img = Image.open("images.jfif")
img = img.resize((100, 100))
gray_img = img.convert("L")
gray_img.save("output_grayscale.png")
target_array = np.array(gray_img, dtype=np.float64) 

# ----------------------------------------------------------------------
# 2. Initialization
# ----------------------------------------------------------------------
pop_size = 90
img_h, img_w = 100, 100
population = np.random.randint(0, 256, (pop_size, img_h, img_w)).astype(np.float64)

# ----------------------------------------------------------------------
# 3. GA functions
# ----------------------------------------------------------------------

def fitness_all(pop, target):

    diff = pop - target            
    mse = np.mean(diff ** 2, axis=(1, 2))   
    return -mse


def crossover_vectorized(parents1, parents2):

    n = parents1.shape[0]
    mask = np.random.rand(n, img_h, img_w) > 0.5
    return np.where(mask, parents1, parents2)


def mutate_vectorized(children, mutation_rate):

    mutation_mask = np.random.rand(*children.shape) < mutation_rate
    n_mutations = np.sum(mutation_mask)
    if n_mutations > 0:
        children[mutation_mask] = np.random.randint(0, 256, n_mutations)
    return children


def tournament_select_indices(fitnesses, n_selections, tournament_size=3):

    pop_size = len(fitnesses)
    # candidates: shape (n_selections, tournament_size)
    candidates = np.random.randint(0, pop_size, size=(n_selections, tournament_size))
    candidate_fitnesses = fitnesses[candidates]              # (n_selections, tournament_size)
    winner_pos = np.argmax(candidate_fitnesses, axis=1)      
    winners = candidates[np.arange(n_selections), winner_pos]
    return winners


# ----------------------------------------------------------------------
# 4. Main GA loop
# ----------------------------------------------------------------------
max_generations = 10000
elitism_count = 2  # number of best individuals carried over unchanged each gen

fitness_history = []
best_fitness_history = []
best_individual = None
best_fitness = -float('inf')



for generation in range(max_generations):
    #  evaluate fitness for entire population
    fitnesses = fitness_all(population, target_array)

    fitness_history.append(np.mean(fitnesses))
    current_best = np.max(fitnesses)
    best_fitness_history.append(current_best)

    current_best_idx = np.argmax(fitnesses)
    if fitnesses[current_best_idx] > best_fitness:
        best_fitness = fitnesses[current_best_idx]
        best_individual = population[current_best_idx].copy()

    # build the WHOLE next generation
    sorted_idx = np.argsort(-fitnesses)  # best first
    new_population = np.empty_like(population)

    # elitism
    new_population[:elitism_count] = population[sorted_idx[:elitism_count]]

    n_children = pop_size - elitism_count

    # select all parents for all children
    parent1_idx = tournament_select_indices(fitnesses, n_children, tournament_size=3)
    parent2_idx = tournament_select_indices(fitnesses, n_children, tournament_size=3)

    parents1 = population[parent1_idx]  
    parents2 = population[parent2_idx]   

    children = crossover_vectorized(parents1, parents2)

    # mutation rate decays
    mutation_rate = 0.05 * (1 - generation / max_generations) + 0.001
    children = mutate_vectorized(children, mutation_rate)

    new_population[elitism_count:] = children
    population = new_population

    if generation % 500 == 0:
        print(f"Gen {generation:5d} | Best Fitness = {current_best:9.2f} | "
              f"Avg = {np.mean(fitnesses):9.2f}")




# ----------------------------------------------------------------------
# 5. Final results
# ----------------------------------------------------------------------
final_image_array = np.clip(best_individual, 0, 255).astype(np.uint8)
final_image = Image.fromarray(final_image_array, mode='L')
final_image.save("output_final.png")

print(f"Final Best Fitness: {best_fitness:.2f}")
print(f"Final MSE: {-best_fitness:.2f}")

# ----------------------------------------------------------------------
# 6. Fitness / MSE history plots
# ----------------------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fitness_history, label='Average Fitness', alpha=0.7)
plt.plot(best_fitness_history, label='Best Fitness', alpha=0.7)
plt.xlabel('Generation')
plt.ylabel('Fitness (-MSE)')
plt.title('Fitness Improvement')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(-np.array(fitness_history), label='Average MSE', alpha=0.7)
plt.plot(-np.array(best_fitness_history), label='Best MSE', alpha=0.7)
plt.xlabel('Generation')
plt.ylabel('MSE')
plt.title('Error Reduction')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png')
print("Saved: output_grayscale.png, output_final.png, training_history.png")