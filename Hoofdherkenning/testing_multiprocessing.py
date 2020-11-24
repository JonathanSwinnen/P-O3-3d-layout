import time
import multiprocessing


def basic_func(x):
    if x == 0:
        return 'zero'
    elif x % 2 == 0:
        return 'even'
    else:
        return 'odd'


def multiprocessing_func(x):
    y = x * x
    time.sleep(2) # this func takes 2s
    print('{} squared results in a/an {} number'.format(x, basic_func(y)))


if __name__ == '__main__':
    print(multiprocessing.cpu_count())
    print("No multiprocessing:")
    starttime = time.time()
    for i in range(0, 10):
        y = i * i
        time.sleep(2) #this func takes 2s
        print('{} squared results in a/an {} number'.format(i, basic_func(y)))

    print('That took {} seconds'.format(time.time() - starttime))

    print("Using multiprocessing:")
    starttime = time.time()
    processes = []
    for i in range(0, 10):
        p = multiprocessing.Process(target=multiprocessing_func, args=(i,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print('That took {} seconds'.format(time.time() - starttime))

