import getopt
import glob
import os
import sys
import sqlite3
import json

holds_dict = {}

def get_latest_db():
    # assign directory
    directory = '*'

    # iterate over files in
    # that directory
    list_of_files = glob.glob(f'{directory}/*.sqlite3*')

    if len(list_of_files) == 0:
        print('Critical Error: No DB file found.')
        sys.exit()

    print('Found input files:')
    for filename in list_of_files:
        print(f'\t{filename}')
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f'Latest file: {latest_file}')
    return latest_file

def hold_role_is_start(role):
    role = str(role)
    if role == '12' or role == '20' or role == '24' or role == '28' or role == '32' or role == '39' or role == '42':
        return True
    else:
        return False

def hold_role_is_hand(role):
    role = str(role)
    if role == '13' or role == '21' or role == '25' or role == '29' or role == '33' or role == '36' or role == '37' or role == '41' or role == '43':
        return True
    else:
        return False

def hold_role_is_finish(role):
    role = str(role)
    if role == '14' or role == '22' or role == '26' or role == '30' or role == '34' or role == '44' or role == '42':
        return True
    else:
        return False

def hold_role_is_feet(role):
    role = str(role)
    if role == '15' or role == '23' or role == '27' or role == '31' or role == '35' or role == '45':
        return True
    else:
        return False

def get_hold_role_string(role):

    if hold_role_is_start(role):
        return "start"
    elif hold_role_is_hand(role):
        return "hand"
    elif hold_role_is_finish(role):
        return "finish"
    elif hold_role_is_feet(role):
        return "foot"
    else:
        return "unknown"

def get_holds_summary(boulder_string):
    # TODO: What are these char for?
    boulder_string = boulder_string.replace('"', '')
    boulder_string = boulder_string.replace(',', '')

    holds_summary = []
    while len(boulder_string) > 0:
        action = boulder_string[0]

        if action == 'p':
            hold_id = boulder_string[1:5]
            hold_role = boulder_string[6:8]

            holds_summary.append({"hold_id": hold_id,
                                  "hold_role_id": hold_role,
                                  "hold_role": get_hold_role_string(hold_role),
                                  "hold_position": holds_dict[hold_id]
                                  })

            boulder_string = boulder_string[8:]
        if action == 'x':
            boulder_string = boulder_string[5:]

    return holds_summary

def get_boulder_object(boulder, boulder_holds):
    boulder_id = boulder[0]

    for b in boulder_holds:
        if boulder_id == b[0]:

            boulder_angle = boulder[1]
            boulder_difficulty = boulder[2]
            boulder_quality = boulder[3]

            boulder_name = b[1]
            boulder_no_match = int("no match" in b[2].lower())
            boulder_holds_raw = b[3]
            boulder_created_at = b[4]

            boulder_holds_summary = get_holds_summary(boulder_holds_raw)

            boulder = {
                "boulder_id": boulder_id,
                "boulder_name": boulder_name,
                "boulder_angle": boulder_angle,
                "boulder_difficulty": boulder_difficulty,
                "boulder_quality": boulder_quality,
                "boulder_no_match": boulder_no_match,
                "boulder_created_at": boulder_created_at,
                "boulder_holds_raw": boulder_holds_raw,
                "boulder_holds": boulder_holds_summary
            }
            return boulder
    return None

def write_to_file(object):
    with open("ddbb.json", "w") as f:
        json.dump(object, f, indent=4, sort_keys=True)

def generate_json_ddbb(database_file):
    try:
        sqlite_connection = sqlite3.connect(database_file)
        cursor = sqlite_connection.cursor()

        # GET position for each board hold and build dictionary:
        sql_query = 'SELECT id, x, y FROM holes;'
        cursor.execute(sql_query)
        holes_position_table = cursor.fetchall()
        print(f'TOTAL hold/holes positions found: {len(holes_position_table)}')

        # Build hold dictionary with Hold ID, Hold X and Hold Y
        holes_dict = {}
        for hole_id, hole_x, hole_y in holes_position_table:
            holes_dict[str(hole_id)] = {
                'x': hole_x,
                'y': hole_y
            }
        holds_position_table = None

        sql_query = 'SELECT id, hole_id FROM placements WHERE layout_id = 1;'
        cursor.execute(sql_query)
        holds_placement_raw = cursor.fetchall()
        print(f'TOTAL holds placement found: {len(holds_placement_raw)}')

        # Add placement information to holds dictionary
        for placement_id, hole_id in holds_placement_raw:
            holds_dict[str(placement_id)] = {"placement_id": placement_id,
                                             "hole_id": hole_id,
                                             "possition": holes_dict[str(hole_id)]}
        holds_placement_raw = None

        # GET relevant BOULDER holds from DDBB:
        sql_query = 'SELECT uuid, name, description, frames, created_at FROM climbs WHERE layout_id = 1 AND frames_count = 1 AND is_draft = 0 AND is_listed = 1 ORDER BY created_at ASC;'
        cursor.execute(sql_query)
        boulder_holds = cursor.fetchall()

        # GET relevant BOULDER grades from DDBB:
        sql_query = 'SELECT climb_uuid, angle, display_difficulty, quality_average FROM climb_stats WHERE ascensionist_count > 20;'
        cursor.execute(sql_query)
        boulder_grades = cursor.fetchall()

        boulder_collection = []

        for boulder in boulder_grades:
            boulder_detailed = get_boulder_object(boulder, boulder_holds)
            if boulder_detailed is None:
                continue
            boulder_collection.append(boulder_detailed)

        write_to_file(boulder_collection)

    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)
    finally:
        if sqlite_connection:
            sqlite_connection.close()
            print("The SQLite connection is closed")


def main(argv, null=None):
    inputfile = ''
    outputfile = ''

    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])

    except getopt.GetoptError:
        print('ERROR (getopt.GetoptError): kilter_nerd.py -i <inputfile> -o <outputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('kilter_nerd.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    if not inputfile:
        print("No input file! looking for latest DDBB")
        inputfile = get_latest_db()

    print('Input file is "', inputfile)
    print('Output file is (not used)"', outputfile)

    generate_json_ddbb(inputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
    print('done')
