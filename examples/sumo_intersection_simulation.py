# sumo_intersection_simulation.py
import os
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import argparse

def prepare_intersection_data_for_sumo(traffic_data_file, output_dir="sumo_files"):
    """
    Prepare TomTom API traffic data for SUMO simulation.
    
    Args:
        traffic_data_file (str): JSON file with traffic data
        output_dir (str): Directory to save SUMO input files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load traffic data
    with open(traffic_data_file, 'r') as f:
        traffic_data = json.load(f)
    
    # Extract intersection geometry
    intersection = traffic_data["intersection"]
    lat = intersection["latitude"]
    lon = intersection["longitude"]
    
    # Create a basic intersection network file
    create_basic_intersection_xml(
        intersection_id="roswell_hwy41",
        lat=lat,
        lon=lon,
        output_dir=output_dir
    )
    
    # Generate traffic demand based on approach data
    approaches = traffic_data["approaches"]
    generate_traffic_demand(approaches, os.path.join(output_dir, "traffic_demand.rou.xml"))
    
    # Generate adaptive traffic light program
    generate_adaptive_tls_program(approaches, os.path.join(output_dir, "traffic_lights.add.xml"))
    
    # Create SUMO configuration file
    create_sumo_config(output_dir)
    
    print(f"Successfully prepared SUMO files in {output_dir}")
    return output_dir

def create_basic_intersection_xml(intersection_id, lat, lon, output_dir):
    """
    Create a basic SUMO intersection XML file.
    
    Args:
        intersection_id (str): Identifier for the intersection
        lat (float): Latitude of intersection center
        lon (float): Longitude of intersection center
        output_dir (str): Output directory for SUMO files
    """
    # Create node definitions
    nodes_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <node id="center" x="0.0" y="0.0" type="traffic_light"/>
    <node id="N" x="0.0" y="100.0" type="priority"/>
    <node id="S" x="0.0" y="-100.0" type="priority"/>
    <node id="E" x="100.0" y="0.0" type="priority"/>
    <node id="W" x="-100.0" y="0.0" type="priority"/>
</nodes>
"""
    
    # Create edge definitions
    edges_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <edge id="NtoCenter" from="N" to="center" priority="1" numLanes="2"/>
    <edge id="CentertoN" from="center" to="N" priority="1" numLanes="2"/>
    <edge id="StoCenter" from="S" to="center" priority="1" numLanes="2"/>
    <edge id="CentertoS" from="center" to="S" priority="1" numLanes="2"/>
    <edge id="EtoCenter" from="E" to="center" priority="1" numLanes="2"/>
    <edge id="CentertoE" from="center" to="E" priority="1" numLanes="2"/>
    <edge id="WtoCenter" from="W" to="center" priority="1" numLanes="2"/>
    <edge id="CentertoW" from="center" to="W" priority="1" numLanes="2"/>
</edges>
"""
    
    # Create connection definitions
    connections_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<connections xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/connections_file.xsd">
    <!-- North to South (straight) -->
    <connection from="NtoCenter" to="CentertoS" fromLane="1" toLane="1"/>
    <!-- North to East (right) -->
    <connection from="NtoCenter" to="CentertoE" fromLane="0" toLane="0"/>
    <!-- North to West (left) -->
    <connection from="NtoCenter" to="CentertoW" fromLane="1" toLane="1"/>
    
    <!-- South to North (straight) -->
    <connection from="StoCenter" to="CentertoN" fromLane="1" toLane="1"/>
    <!-- South to West (right) -->
    <connection from="StoCenter" to="CentertoW" fromLane="0" toLane="0"/>
    <!-- South to East (left) -->
    <connection from="StoCenter" to="CentertoE" fromLane="1" toLane="1"/>
    
    <!-- East to West (straight) -->
    <connection from="EtoCenter" to="CentertoW" fromLane="1" toLane="1"/>
    <!-- East to North (right) -->
    <connection from="EtoCenter" to="CentertoN" fromLane="0" toLane="0"/>
    <!-- East to South (left) -->
    <connection from="EtoCenter" to="CentertoS" fromLane="1" toLane="1"/>
    
    <!-- West to East (straight) -->
    <connection from="WtoCenter" to="CentertoE" fromLane="1" toLane="1"/>
    <!-- West to South (right) -->
    <connection from="WtoCenter" to="CentertoS" fromLane="0" toLane="0"/>
    <!-- West to North (left) -->
    <connection from="WtoCenter" to="CentertoN" fromLane="1" toLane="1"/>
</connections>
"""
    
    # Create traffic light definitions (default program)
    tls_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    <tlLogic id="center" type="static" programID="0" offset="0">
        <phase duration="31" state="GGggrrrrGGggrrrr"/> <!-- North-South green -->
        <phase duration="4" state="yyggrrrryyggRRRR"/> <!-- North-South yellow -->
        <phase duration="31" state="rrrrGGggrrrrGGgg"/> <!-- East-West green -->
        <phase duration="4" state="rrrryyggrrrryygg"/> <!-- East-West yellow -->
    </tlLogic>
</additional>
"""
"""
    
    # Write files
    with open(os.path.join(output_dir, f"{intersection_id}.nod.xml"), 'w') as f:
        f.write(nodes_xml)
    
    with open(os.path.join(output_dir, f"{intersection_id}.edg.xml"), 'w') as f:
        f.write(edges_xml)
    
    with open(os.path.join(output_dir, f"{intersection_id}.con.xml"), 'w') as f:
        f.write(connections_xml)
    
    with open(os.path.join(output_dir, f"{intersection_id}.tll.xml"), 'w') as f:
        f.write(tls_xml)
    
    # Use SUMO's netconvert to generate the final network file
    try:
        subprocess.run([
            "netconvert",
            "--node-files", os.path.join(output_dir, f"{intersection_id}.nod.xml"),
            "--edge-files", os.path.join(output_dir, f"{intersection_id}.edg.xml"),
            "--connection-files", os.path.join(output_dir, f"{intersection_id}.con.xml"),
            "--tllogic-files", os.path.join(output_dir, f"{intersection_id}.tll.xml"),
            "--output-file", os.path.join(output_dir, f"{intersection_id}.net.xml")
        ], check=True)
        print(f"Successfully created SUMO network file: {os.path.join(output_dir, f'{intersection_id}.net.xml')}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating network file: {e}")
    except FileNotFoundError:
        print("netconvert command not found. Make sure SUMO is installed and in your PATH.")
    
    return os.path.join(output_dir, f"{intersection_id}.net.xml")

def generate_traffic_demand(approaches, output_file):
    """
    Generate SUMO traffic demand (.rou.xml) from TomTom API traffic data.
    
    Args:
        approaches (dict): Processed traffic data from TomTom API
        output_file (str): Path to output .rou.xml file
    """
    # Start building the routes XML
    routes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Define vehicle types -->
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="55.55" color="1,1,0"/>
    
    <!-- Define routes -->
"""
    
    # Define directions and their corresponding routes
    directions = {
        "North": {"from": "N", "to": "S", "route_id": "north_south"},
        "South": {"from": "S", "to": "N", "route_id": "south_north"},
        "East": {"from": "E", "to": "W", "route_id": "east_west"},
        "West": {"from": "W", "to": "E", "route_id": "west_east"}
    }
    
    # Add route definitions
    for direction, route_info in directions.items():
        routes_xml += f'    <route id="{route_info["route_id"]}" edges="{route_info["from"]}toCenter Centerto{route_info["to"]}"/>\n'
    
    # Add flows based on approach data
    for direction, data in approaches.items():
        if "error" in data:
            continue
            
        # Calculate flow rate based on speed and congestion
        speed_ratio = data.get("speed_ratio", 0.8)
        current_speed = data.get("current_speed", 30)
        
        # Approximate vehicle count (cars per hour)
        # Lower speeds with low speed_ratio indicate higher volume
        if speed_ratio < 0.5:
            # Heavy congestion
            vph = 1200
        elif speed_ratio < 0.75:
            # Moderate congestion
            vph = 800
        else:
            # Light congestion
            vph = 400
        
        # Add flow to XML if direction is in our mapping
        if direction in directions:
            route_id = directions[direction]["route_id"]
            routes_xml += f"""    <flow id="flow_{direction.lower()}" route="{route_id}" begin="0" end="3600" vehsPerHour="{vph}" type="car"/>\n"""
    
    # Close the XML
    routes_xml += "</routes>"
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(routes_xml)
    
    print(f"Successfully created traffic demand file: {output_file}")
    return output_file

def generate_adaptive_tls_program(approaches, output_file):
    """
    Generate adaptive traffic light program based on approach traffic conditions.
    
    Args:
        approaches (dict): Processed traffic data from TomTom API
        output_file (str): Path to output .add.xml file with traffic light logic
    """
    # Calculate congestion levels for each approach direction
    north_south_congestion = 0
    east_west_congestion = 0
    
    # Get congestion levels for each approach
    direction_count = {"ns": 0, "ew": 0}
    
    for direction, data in approaches.items():
        if "error" in data:
            continue
            
        speed_ratio = data.get("speed_ratio", 1.0)
        congestion = 1.0 - speed_ratio  # Convert speed ratio to congestion (0-1)
        
        if direction in ["North", "South"]:
            north_south_congestion += congestion
            direction_count["ns"] += 1
        elif direction in ["East", "West"]:
            east_west_congestion += congestion
            direction_count["ew"] += 1
    
    # Average the congestion for each axis (if data is available)
    if direction_count["ns"] > 0:
        north_south_congestion /= direction_count["ns"]
    if direction_count["ew"] > 0:
        east_west_congestion /= direction_count["ew"]
    
    # Calculate green times (30-60 seconds range)
    min_green = 30
    max_green = 60
    range_green = max_green - min_green
    
    total_congestion = north_south_congestion + east_west_congestion
    if total_congestion > 0:
        # Proportional allocation based on congestion
        ns_proportion = north_south_congestion / total_congestion
        ew_proportion = east_west_congestion / total_congestion
    else:
        # Equal split if no congestion data
        ns_proportion = 0.5
        ew_proportion = 0.5
    
    # Calculate green times
    ns_green = min_green + int(range_green * ns_proportion)
    ew_green = min_green + int(range_green * ew_proportion)
    
    # Standard yellow time
    yellow_time = 4
    
    # Create traffic light program XML
    tls_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    <tlLogic id="center" type="static" programID="adaptive" offset="0">
        <phase duration="{ns_green}" state="GGrrGGrr"/> <!-- North-South green -->
        <phase duration="{yellow_time}" state="yyrryyrr"/> <!-- North-South yellow -->
        <phase duration="{ew_green}" state="rrGGrrGG"/> <!-- East-West green -->
        <phase duration="{yellow_time}" state="rryyrryy"/> <!-- East-West yellow -->
    </tlLogic>
</additional>
"""
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(tls_xml)
    
    print(f"Successfully created adaptive traffic light program: {output_file}")
    print(f"Green times: North-South = {ns_green}s, East-West = {ew_green}s")
    
    return {
        "north_south_green": ns_green,
        "east_west_green": ew_green,
        "yellow_time": yellow_time
    }

def create_sumo_config(output_dir):
    """
    Create SUMO configuration file (.sumocfg)
    
    Args:
        output_dir (str): Directory to save the configuration file
    """
    config_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="roswell_hwy41.net.xml"/>
        <route-files value="traffic_demand.rou.xml"/>
        <additional-files value="traffic_lights.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="0.1"/>
    </time>
    <processing>
        <time-to-teleport value="300"/>
    </processing>
    <report>
        <verbose value="true"/>
        <duration-log.statistics value="true"/>
    </report>
    <gui_only>
        <gui-settings-file value="gui-settings.xml"/>
    </gui_only>
</configuration>
"""
    
    # Write the config file
    with open(os.path.join(output_dir, "roswell_hwy41.sumocfg"), 'w') as f:
        f.write(config_xml)
    
    # Create a GUI settings file for better visualization
    gui_settings = f"""<?xml version="1.0" encoding="UTF-8"?>
<viewsettings>
    <scheme name="real world"/>
    <delay value="50"/>
    <viewport zoom="100" x="0" y="0"/>
    <decal file="background.jpg" centerX="0" centerY="0" width="1000" height="1000" rotation="0.00"/>
</viewsettings>
"""
    
    with open(os.path.join(output_dir, "gui-settings.xml"), 'w') as f:
        f.write(gui_settings)
    
    print(f"Successfully created SUMO configuration file: {os.path.join(output_dir, 'roswell_hwy41.sumocfg')}")

def compare_fixed_vs_adaptive(output_dir):
    """
    Run both fixed and adaptive traffic light simulations and compare results
    
    Args:
        output_dir (str): Directory containing SUMO files
    """
    results = {
        "fixed": {},
        "adaptive": {}
    }
    
    # Run simulations for both fixed and adaptive traffic lights
    for program_id in ["0", "adaptive"]:
        program_type = "fixed" if program_id == "0" else "adaptive"
        
        try:
            # Run SUMO in command-line mode to get statistics
            result = subprocess.run([
                "sumo", 
                "-c", os.path.join(output_dir, "roswell_hwy41.sumocfg"),
                "--tlslogic.program", program_id,
                "--duration-log.statistics",
                "--log", os.path.join(output_dir, f"{program_type}_log.txt"),
                "--statistic-output", os.path.join(output_dir, f"{program_type}_stats.xml")
            ], capture_output=True, text=True, check=True)
            
            # Parse the output to extract metrics
            output_lines = result.stdout.split('\n')
            
            for line in output_lines:
                if "Statistics (avg):" in line:
                    metrics_line = output_lines[output_lines.index(line) + 1]
                    metrics = metrics_line.split()
                    
                    results[program_type] = {
                        "waiting_time": float(metrics[1]),
                        "time_loss": float(metrics[3]),
                        "duration": float(metrics[5])
                    }
            
            print(f"Successfully ran {program_type} traffic light simulation")
            
        except subprocess.CalledProcessError as e:
            print(f"Error running {program_type} simulation: {e}")
    
    # Compare results
    if results["fixed"] and results["adaptive"]:
        waiting_time_imp = ((results["fixed"]["waiting_time"] - results["adaptive"]["waiting_time"]) 
                           / results["fixed"]["waiting_time"]) * 100
        time_loss_imp = ((results["fixed"]["time_loss"] - results["adaptive"]["time_loss"]) 
                        / results["fixed"]["time_loss"]) * 100
        
        print("\n=== SIMULATION RESULTS COMPARISON ===")
        print(f"Fixed timing - Avg waiting time: {results['fixed']['waiting_time']:.2f}s, Time loss: {results['fixed']['time_loss']:.2f}s")
        print(f"Adaptive timing - Avg waiting time: {results['adaptive']['waiting_time']:.2f}s, Time loss: {results['adaptive']['time_loss']:.2f}s")
        print(f"Improvement - Waiting time: {waiting_time_imp:.2f}%, Time loss: {time_loss_imp:.2f}%")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Generate SUMO simulation from TomTom traffic data')
    parser.add_argument('--data', required=True, help='Path to TomTom traffic data JSON file')
    parser.add_argument('--output', default='sumo_files', help='Output directory for SUMO files')
    parser.add_argument('--compare', action='store_true', help='Run and compare fixed vs adaptive traffic light simulations')
    args = parser.parse_args()
    
    # Prepare the simulation files
    output_dir = prepare_intersection_data_for_sumo(args.data, args.output)
    
    # Run comparison if requested
    if args.compare:
        results = compare_fixed_vs_adaptive(output_dir)
    
    print(f"""
SUMO simulation files created in {output_dir}

To run the simulation with SUMO-GUI:
    sumo-gui -c {os.path.join(output_dir, "roswell_hwy41.sumocfg")}

To run with the adaptive traffic light program:
    sumo-gui -c {os.path.join(output_dir, "roswell_hwy41.sumocfg")} --tlslogic.program adaptive

To run a headless simulation:
    sumo -c {os.path.join(output_dir, "roswell_hwy41.sumocfg")}
""")

if __name__ == "__main__":
    main()