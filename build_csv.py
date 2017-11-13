#!/usr/bin/env python

import csv
import fileinput
import re

def parse():
    result = None
    results = []
    for line in fileinput.input():
        if not line.strip():
            continue
        if line.startswith('Expansions'):
            continue
        match = re.match(r'Solving Air Cargo Problem (.) using (.*)\.\.\..*', line)
        if match:
            if result:
                result['solution'] = ' '.join(solution)
                results.append(result)
            result = {'problem': match.group(1), 'method': match.group(2)}
            solution = []
            continue
        match = re.match(r'\s*([0-9]+)\s+', line)
        if match:
            result['expansions'] = match.group(1)
            continue
        match = re.match(r'Plan length: ([0-9]+) .*: ([0-9.]+)', line)
        if match:
            result['plan_length'] = match.group(1)
            result['time'] = '%.2f' % float(match.group(2))
            continue
        solution.append(line.strip())
    if result:
        result['solution'] = ' '.join(solution)
        results.append(result)


    with open('/dev/stdout', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parse()
