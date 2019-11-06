from PIL import Image
import numpy as np
from algorithms_clean import Algorithm, SA
import time
import os
import csv
from multiprocessing import Process, current_process

# genome size settings
polygons = 250
vertices = 1000

def experiment(name, algorithm, paintings, repetitions, polys, iterations, savepoints):
    # get date/time
    now = time.strftime("%c")
    new_name = "Experiments/" + name
    # create experiment directory with log .txt file
    if not os.path.exists(new_name):
        os.makedirs(new_name)

    total_runs = len(polys) * len(paintings) * repetitions

    # logging a lot of metadata
    logfile = name+"/"+name+"-LOG.txt"
    with open(logfile, 'a') as f:
        f.write("EXPERIMENT " + name + " LOG\n")
        f.write("DATE " + now + "\n\n")
        f.write("STOP CONDITION " +str(iterations)+ " iterations\n\n")
        f.write("LIST OF PAINTINGS (" + str(len(paintings)) +")\n")
        for painting in paintings:
            f.write(painting + "\n")
        f.write("\n")
        f.write("POLYS " + str(len(polys)) + " " + str(polys) + "\n\n")
        f.write("REPETITIONS " +str(repetitions) + "\n\n")
        f.write("RESULTING IN A TOTAL OF " + str(total_runs) + " RUNS\n\n")
        f.write("STARTING EXPERIMENT NOW!\n")

    # initializing the main datafile
    datafile = name+"/"+name + "-DATA.csv"
    header = ["Painting", "Vertices", " Replication", "MSE"]
    with open(datafile, 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # main experiment, looping through repetitions, poly numbers, and paintings:
    exp = 1
    for painting in paintings:
        painting_name = painting.split("/")[1].split("-")[0]
        for poly in polys:
            for repetition in range(repetitions):
                tic = time.time()
                # make a directory for this run, containing the per iteration data and a selection of images
                outdir = name + "/" + str(exp) + "-" + str(repetition) + "-" + str(poly) + "-" + painting_name
                os.makedirs(outdir)

                # Set image in np values
                im_goal = Image.open(painting)
                goal = np.array(im_goal)
                h, w = np.shape(goal)[0], np.shape(goal)[1]

                # Run the simulated annealing
                solver = SA(goal, w, h, poly, poly * 4, "MSE", savepoints, outdir, iterations)
                solver.run()
                solver.write_data()
                bestMSE = solver.best.fitness

                # save best value in maindata sheet
                datarow = [painting_name, str(poly * 4), str(repetition), bestMSE]

                with open(datafile, 'a', newline = '') as f:
                    writer = csv.writer(f)
                    writer.writerow(datarow)

                toc = time.time()
                now = time.strftime("%c")
                with open(logfile, 'a') as f:
                    f.write(now + " finished run " + str(exp) + "/" + str(total_runs) + " n: " + str(repetition) + " poly: " + str(poly) + " painting: " + painting_name + " in " + str((toc - tic)/60) + " minutes\n")

                exp += 1

name = "1miltest.x2"
paintings = ["paintings/monalisa-240-180.png", "paintings/bach-240-180.png", "paintings/dali-240-180.png", "paintings/mondriaan2-180-240.png", "paintings/pollock-240-180.png", "paintings/starrynight-240-180.png", "paintings/kiss-180-240.png"]
paintings = ["paintings/mondriaan2-180-240.png"]
# Define a list of savepoints, more in the first part of the run, and less later.
savepoints = list(range(0, 250000, 1000)) + list(range(250000, 1000000, 10000))
repetitions = 2
polys = [250]
iterations = 10000

# Experiment name.
names = ["c50_10ki_250poly_path_test"]

if __name__ == '__main__':
    worker_count = 1
    for i in range(worker_count):
        args = (names[i], "SA", paintings, repetitions, polys, iterations, savepoints)
        p = Process(target=experiment, args=args)
        p.start()
