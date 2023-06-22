import evacuate
import sys

def main(nr_runs, follower_rate, scaredness_rate):
    stats = ""
    for i in range(nr_runs):
        stats += evacuate.main(['--input', 'in/merc.txt', '--numpeople', '100', '--max_time', '250', '--scaredness_rate', str(scaredness_rate), '--follower_rate', str(follower_rate), '--run_id', str(i)])
    
    print(stats)

if __name__ == '__main__':
    nr_runs = 1
    nr_runs = int(sys.argv[1])
    follower_rate = float(sys.argv[2])
    scaredness_rate = float(sys.argv[3])
    main(nr_runs, follower_rate, scaredness_rate)
