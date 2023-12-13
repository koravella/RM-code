import itertools as it
import numpy as np
import argparse
import math

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('r', type=int, help='r in RM(r, m)')
    parser.add_argument('m', type=int, help='m in RM(r, m)')
    parser.add_argument('--show_gen_matrix', '-s', action='store_true',
                        help='Show generating matrix of the RM(r, m)')
    parser.add_argument('--encode', '-e', type=str, help='Vector for encoding')
    parser.add_argument('--decode', '-d', type=str, help='Vector for encoding')
    parser.add_argument('--show_num_errors', '-t', action='store_true',
                        help='Show number of possible errors for RM(r,m)')
    return parser

def get_generating_matrix(r, m):
    num_rows = sum([math.comb(m, i) for i in range(r + 1)])
    num_cols = 2**m
    G = np.empty(shape = [num_rows, num_cols], dtype=np.uint8)
    
    cur_row = 0
    for i in range(r + 1):
        cmbs = it.combinations(range(1, m + 1), i)
        for cmb in cmbs:
            x = 0
            for k in cmb:
                x += 1 << (m - k)
            
            for cur_col in range(num_cols):
                G[cur_row][cur_col] = (x & cur_col) == x
            
            cur_row += 1
    return G

def transform_encode(vect, r, m):
    size = sum([math.comb(m, i) for i in range(r + 1)])
    if len(vect) != size:
        print('Wrong lenght of the vector to encode')
        exit(1)
    
    if not all((x == '0' or x == '1') for x in vect):
        print('The vector to encode must consist of 0 and 1')
        exit(1)
    
    return np.array([int(x) - int('0') for x in vect], dtype=np.uint8)

def transform_decode(vect, m):
    size = 2**m
    if len(vect) != size:
        print('Wrong lenght of the vector to decode')
        exit(1)
    
    if not all((x == '0' or x == '1') for x in vect):
        print('The vector to decode must consist of 0 and 1')
        exit(1)
    
    return np.array([int(x) - int('0') for x in vect], dtype=np.uint8)

def encode(G, vencode):
    return np.dot(vencode, G) % 2

def decode(r, m, G, vdecode):
    size = sum([math.comb(m, i) for i in range(r + 1)])
    u = np.zeros(size, dtype = np.uint8)
    vect = vdecode.copy()

    for i in range(r, -1, -1):
        cmbs = it.combinations(range(1, m + 1), i)
        cur = sum([math.comb(m, s) for s in range(i)])

        for cmb in cmbs:
            x = 0
            for k in cmb:
                x += 1 << (m - k)
            mask = ~x & ((1 << m) - 1)
                
            arr = np.zeros(2**m, dtype = np.uint8)
            for k in range(2**m):
                ind = k & mask
                arr[ind] =  arr[ind] ^ vect[k]
            N_ones = sum(arr)
            N_zeros = 2**(m - i) - N_ones
            
            u[cur] = 1 if N_ones > N_zeros else 0
            #print(f"u={u} cur={cur} mask={mask} N_ones={N_ones} N_zeros={N_zeros}")
            cur += 1
        vect = (vdecode + np.dot(u, G)) % 2
    return u

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    
    if args.encode != None:
        vencode = transform_encode(args.encode, args.r, args.m)
    if args.decode != None:
        vdecode = transform_decode(args.decode, args.m)
    
    if args.show_num_errors:
        num_err = (2**(args.m - args.r) - 1) // 2
        print(f"\nCorrect {num_err} errors")
    
    G = get_generating_matrix(args.r, args.m)
    if args.show_gen_matrix:
        print("\nGenerating matrix G:")
        print(G)
    
    if args.encode != None:
        print(f"\nEncoding vector:")
        print(''.join([chr(x+ord('0')) for x in encode(G, vencode)]))

    if args.decode != None:
        print(f"\nDecoding vector:") 
        print(''.join([chr(x+ord('0')) for x in decode(args.r, args.m, G, vdecode)]))
        
main()
