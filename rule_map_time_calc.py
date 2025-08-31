import re
from datetime import datetime

log_path = "rule_map_time_calc.log"
start_pattern = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - rule_mapper - INFO - Processing row")
end_pattern = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - rule_mapper - INFO - Analysis completed successfully")

start_times = []
end_times = []

with open(log_path) as f:
    for line in f:
        m_start = start_pattern.match(line)
        m_end = end_pattern.match(line)
        if m_start:
            start_times.append(datetime.strptime(
                m_start.group(1), "%Y-%m-%d %H:%M:%S,%f"))
        if m_end:
            end_times.append(datetime.strptime(
                m_end.group(1), "%Y-%m-%d %H:%M:%S,%f"))

flows = min(5, min(len(start_times), len(end_times)))
durations = [(end_times[i] - start_times[i]).total_seconds()
             for i in range(flows)]
average = sum(durations) / flows

print(f"Average time to process {flows} flows: {average:.2f} seconds")
