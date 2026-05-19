import pickle
import main
import interactions
import visualizer
import sys


if __name__ == "__main__":
    identifier = sys.argv[1]
    outerstep = int(sys.argv[2])
    innerstep = int(sys.argv[3])
    with open(f"data/initial_conditions_{identifier}.pkl", "rb") as f:
        init = pickle.load(f)
        print("loaded")
        data = main.run(
            init["positions"],
            init["velocities"],
            init["mass"],
            67,
            "",
            3 * 15 * 30,
            outerstep,
            innerstep,
            interactions.get_forces,
        )

    with open(f"data/test_run_{identifier}.pkl", "wb") as g:
        pickle.dump(data, g, protocol=pickle.HIGHEST_PROTOCOL)

    visualizer.display_energy(
        data["Kinetic Energy"], data["Potential Energy"], identifier
    )

    visualizer.display(data["Position"], True, identifier)
