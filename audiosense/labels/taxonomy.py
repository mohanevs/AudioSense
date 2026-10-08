"""
AudioSet / AudioSense Label Taxonomy and Category Mapping.
"""

from typing import Dict, List, Optional, Set

# Standard AudioSet Top-Level Categories
CATEGORIES: Dict[str, List[str]] = {
    "Human sounds": [
        "Speech", "Male speech, man speaking", "Female speech, woman speaking",
        "Child speech, kid speaking", "Conversation", "Narration, monologue",
        "Babbling", "Speech synthesizer", "Shout", "Bellow", "Whoop", "Yell",
        "Battle cry", "Children shouting", "Screaming", "Whispering", "Laughter",
        "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle",
        "Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan",
        "Sigh", "Singing", "Choir", "Yodeling", "Chant", "Mantra",
        "Male singing", "Female singing", "Child singing", "Synthetic singing",
        "Rapping", "Humming", "Groan", "Grunt", "Whistling", "Breathing",
        "Wheeze", "Snoring", "Gasp", "Pant", "Snort", "Cough",
        "Throat clearing", "Sneeze", "Sniff", "Run", "Shuffle",
        "Walk, footsteps", "Chewing, mastication", "Biting", "Gargling",
        "Stomach rumble", "Burping, eructation", "Hiccup", "Fart", "Hands",
        "Finger snapping", "Clapping", "Heart sounds, heartbeat", "Heart murmur",
        "Cheering", "Applause", "Chatter", "Crowd", "Hubbub, speech noise, speech babble",
        "Children playing"
    ],
    "Animal sounds": [
        "Animal", "Domestic animals, pets", "Dog", "Bark", "Yip", "Howl",
        "Bow-wow", "Growling", "Whimper (dog)", "Cat", "Purr", "Meow",
        "Hiss", "Caterwaul", "Livestock, farm animals", "Horse", "Clip-clop",
        "Neigh, whinny", "Cattle, bovinae", "Moo", "Cowbell", "Pig", "Oink",
        "Goat", "Bleat", "Sheep", "Fowl", "Chicken, rooster", "Cluck",
        "Crowing, cock-a-doodle-doo", "Turkey", "Gobble", "Duck", "Quack",
        "Goose", "Honk", "Wild animals", "Roar", "Bird", "Bird vocalization, bird call, bird song",
        "Chirp, tweet", "Squawk", "Pigeon, dove", "Coo", "Crow", "Caw", "Owl",
        "Hoot", "Bird flight, flapping wings", "Canidae, dogs, wolves", "Rodents, rats, mice",
        "Mouse", "Patter", "Insect", "Cricket", "Mosquito", "Fly, housefly",
        "Buzz", "Bee, wasp, etc.", "Frog", "Croak", "Snake", "Rattle"
    ],
    "Music": [
        "Music", "Musical instrument", "Plucked string instrument", "Guitar",
        "Electric guitar", "Bass guitar", "Acoustic guitar", "Steel guitar, slide guitar",
        "Tapping (guitar technique)", "Strum", "Banjo", "Sitar", "Mandolin",
        "Zither", "Ukulele", "Keyboard (musical)", "Piano", "Electric piano",
        "Organ", "Electronic organ", "Hammond organ", "Synthesizer", "Sampler",
        "Harpsichord", "Percussion", "Drum kit", "Drum machine", "Drum",
        "Snare drum", "Rimshot", "Drum roll", "Bass drum", "Timpani",
        "Tabla", "Cymbal", "Hi-hat", "Crash cymbal", "Cowbell", "Tambourine",
        "Rattle (instrument)", "Maraca", "Gong", "Tubular bells", "Mallet percussion",
        "Marimba, xylophone", "Glockenspiel", "Vibraphone", "Steelpan",
        "Wind instrument, woodwind", "Flute", "Saxophone", "Clarinet",
        "Harp", "Bell", "Church bell", "Chime", "Wind chime", "Singing bowl"
    ],
    "Natural sounds": [
        "Wind", "Rustling leaves", "Wind noise (microphone)", "Thunderstorm",
        "Thunder", "Water", "Rain", "Raindrop", "Rain on surface", "Stream, burble",
        "Waterfall", "Ocean", "Waves, surf", "Steam", "Gush", "Fire", "Crackle"
    ],
    "Sounds of things": [
        "Vehicle", "Motor vehicle (road)", "Car", "Car passing by", "Race car, auto racing",
        "Car alarm", "Power windows", "Skidding", "Tire squeal", "Car horn",
        "Toot", "Bicycle", "Skateboard", "Bus", "Emergency vehicle",
        "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)",
        "Motorcycle", "Traffic noise, roadway noise", "Rail transport", "Train",
        "Train horn", "Train wheels squealing", "Subway, metro, underground",
        "Aircraft", "Aircraft engine", "Jet engine", "Propeller, airscrew",
        "Helicopter", "Watercraft", "Rowboat, canoe, kayak", "Motorboat, speedboat",
        "Ship", "Engine", "Light engine", "Heavy engine", "Idling", "Accelerating, revving",
        "Door", "Doorbell", "Ding-dong", "Sliding door", "Slam", "Knock",
        "Tap", "Squeak", "Cupboard open or close", "Drawer open or close",
        "Dishes, pots, and pans", "Cutlery, silverware", "Glass", "Chink, clink",
        "Shatter", "Keys jangling", "Clock", "Tick", "Alarm clock", "Telephone",
        "Telephone bell ringing", "Ringtone", "Alarm", "Siren", "Smoke detector, smoke alarm",
        "Fire alarm", "Foghorn", "Whistle", "Steam whistle", "Explosion",
        "Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire",
        "Cap gun", "Fireworks", "Burst, pop", "Eruption", "Boom", "Tools",
        "Hammer", "Jackhammer", "Sawing", "Drill", "Power tool"
    ],
    "Channel, environment and background": [
        "Silence", "Sine wave", "Harmonic", "Chirp tone", "Sound effect",
        "White noise", "Pink noise", "Static", "Hum", "Mains hum",
        "Distortion", "Echo", "Reverberation"
    ]
}

def get_category_for_label(label: str) -> str:
    """Find which category a given label belongs to."""
    label_norm = label.lower().strip()
    for cat, members in CATEGORIES.items():
        for m in members:
            if m.lower() == label_norm or m.lower() in label_norm or label_norm in m.lower():
                return cat
    return "Other environmental sounds"
