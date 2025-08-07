"""Test script to verify room name is properly preserved"""

import json

# Load phase0 result
with open('output/phase0_result.json', 'r', encoding='utf-8') as f:
    phase0_data = json.load(f)

print("Phase 0 Data Structure:")
print(f"- Total items in data array: {len(phase0_data['data'])}")

for i, item in enumerate(phase0_data['data']):
    if i == 0:
        print(f"  [0]: Project Info - Jobsite: {item.get('Jobsite', 'N/A')}")
    else:
        location = item.get('location', 'Unknown')
        rooms = item.get('rooms', [])
        print(f"  [{i}]: {location} - {len(rooms)} rooms")
        for room in rooms:
            print(f"      - {room.get('name', 'MISSING NAME')}")

print("\nExtracted Rooms for Processing:")
all_rooms = []
for item in phase0_data['data']:
    if isinstance(item, dict) and 'rooms' in item:
        for room in item['rooms']:
            room_name = room.get('name', 'MISSING')
            all_rooms.append(room_name)
            print(f"  - Room: {room_name}")

print(f"\nTotal rooms found: {len(all_rooms)}")

# Check if room names are being extracted properly
if 'Unknown' in all_rooms or 'MISSING' in all_rooms:
    print("❌ Error: Room names not properly extracted!")
else:
    print("✅ All room names extracted successfully")