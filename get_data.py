import getopt
import glob
import os
import sys
import sqlite3
import json

def get_latest_db():
    list_of_files = glob.glob('*/*.sqlite3*')
    if not list_of_files:
        print('Critical Error: No DB file found.')
        sys.exit()
    return max(list_of_files, key=os.path.getctime)

def extract_frame_and_grade(database_file):
    """Extracts frame sequences and grades from the database"""
    try:
        conn = sqlite3.connect(database_file)
        cursor = conn.cursor()
        
        # Query to get frames (climbs) and grades (stats) in one join
        query = """
        SELECT c.frames, cs.display_difficulty
        FROM climbs c
        JOIN climb_stats cs ON c.uuid = cs.climb_uuid
        WHERE c.layout_id = 1 
        AND c.frames_count = 1 
        AND c.is_draft = 0 
        AND c.is_listed = 1
        AND cs.ascensionist_count > 20
        ORDER BY c.created_at ASC;
        """
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Format as list of {frame, grade} dictionaries
        data = [{"frame": frame, "grade": grade} for frame, grade in results]
        
        return data
        
    except sqlite3.Error as error:
        print("Database error:", error)
        return []
    finally:
        if conn:
            conn.close()

def write_to_file(data, filename="training_data.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def main(argv):
    inputfile = ''
    
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('Usage: script.py -i <inputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('script.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg

    if not inputfile:
        print("No input file specified, using latest database")
        inputfile = get_latest_db()

    print(f'Processing database: {inputfile}')
    training_data = extract_frame_and_grade(inputfile)
    write_to_file(training_data)
    print(f'Successfully extracted {len(training_data)} entries to training_data.json')

if __name__ == "__main__":
    main(sys.argv[1:])