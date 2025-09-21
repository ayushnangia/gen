# dialogue_generation/config/constants.py
"""
Constants extracted from serial_gen.py and parallel_gen.py
EXACT extraction with no modifications to preserve functionality
"""

from typing import Dict, List, Tuple

# Emotion lists - extracted from both files (lines 99-135)
USER_EMOTION_LIST = [
    "Frustrated", "Angry", "Confused", "Worried", "Disappointed",
    "Happy", "Anxious", "Impatient", "Skeptical", "Desperate",
    "Overwhelmed", "Hopeful", "Satisfied", "Stressed", "Suspicious",
    "Tired", "Excited", "Indifferent", "Grateful", "Demanding"
]

ASSISTANT_EMOTION_LIST = [
    "Professional", "Informative", "Reassuring", "Diplomatic", "Patient",
    "Efficient", "Accommodating", "Solution-focused", "Methodical", "Proactive",
    "Analytical", "Composed", "Detail-oriented", "Responsive", "Thorough",
    "Systematic", "Precise", "Objective", "Resourceful", "Knowledgeable"
]

# Scenario categories - extracted from both files (lines 137-235)
SCENARIO_CATEGORIES = {
    # General categories (applicable across services)
    "general": [
        "account_management",      # Managing user accounts, passwords, etc.
        "cancellation_general",    # General cancellation requests not tied to a specific service
        "complaint",               # General complaints about services or experiences
        "refund_request_general",  # Refunds not specific to a service
        "payment_issues",          # Issues related to payments across services
        "general_inquiry",         # General questions not specific to any service
        "feedback",                # Providing feedback about services or experiences
        "technical_support",       # Technical assistance for app or website issues
        "lost_and_found_general"   # Reporting lost items not tied to a specific service
    ],
    # Restaurant-specific
    "restaurant": [
        "dining_reservation",
        "dietary_requirements",
        "table_modification",
        "special_occasion",
        "menu_inquiry",            # Inquiring about menu items or specials
        "order_status",            # Checking the status of an order
        "reservation_cancellation" # Specific cancellation for reservations
    ],
    # Hotel-specific
    "hotel": [
        "room_reservation",
        "check_in_out",
        "amenity_inquiry",
        "room_service",
        "booking_modification",    # Modifying existing bookings
        "housekeeping_request",    # Requests related to room cleaning and maintenance
        "reservation_cancellation" # Specific cancellation for hotel bookings
    ],
    # Train-specific
    "train": [
        "journey_planning",
        "schedule_inquiry",
        "ticket_booking",
        "platform_information",
        "ticket_change",           # Changing ticket details
        "ticket_cancellation",     # Specific cancellation for train tickets
        "seat_selection"           # Selecting or changing seats
    ],
    # Attraction-specific
    "attraction": [
        "ticket_availability",
        "opening_hours",
        "guided_tour",
        "venue_information",
        "group_booking",           # Booking for groups
        "ticket_cancellation",     # Specific cancellation for attraction tickets
        "accessibility_inquiry"    # Inquiries about accessibility features
    ],
    # Taxi-specific
    "taxi": [
        "ride_booking",
        "pickup_location",
        "fare_inquiry",
        "driver_tracking",
        "ride_cancellation",       # Specific cancellation for taxi rides
        "ride_feedback",           # Providing feedback on taxi rides
        "service_type_inquiry"    # Inquiring about different types of taxi services
    ],
    # Hospital-specific
    "hospital": [
        "appointment_booking",
        "department_inquiry",
        "medical_information",
        "emergency_services",
        "appointment_cancellation",# Specific cancellation for appointments
        "insurance_inquiry",       # Questions about insurance coverage
        "medical_record_request"   # Requesting medical records
    ],
    # Bus-specific
    "bus": [
        "route_information",
        "schedule_inquiry",
        "ticket_booking",
        "stop_location",
        "ticket_change",           # Changing bus ticket details
        "ticket_cancellation",     # Specific cancellation for bus tickets
        "seat_selection"           # Selecting or changing bus seats
    ],
    "flight": [
        "flight_booking",
        "cancellation_flight",     # Specific cancellation for flights
        "ticket_change",
        "baggage_inquiry",
        "check_in",
        "seat_selection",
        "flight_status",
        "upgrade_request",
        "refund_request_flight",   # Refund requests specific to flights
        "lounge_access",
        "boarding_pass_issue",     # Issues related to boarding passes
        "special_meals",           # Requests for special meals on flights
        "pet_transportation"      # Inquiries about transporting pets
    ]
}

# Predefined regions - extracted from both files (exact copy)
PREDEFINED_REGIONS = [
    "Tokyo", "Delhi", "Shanghai", "Sao Paulo", "Mumbai",
    "Beijing", "Cairo", "Mexico City", "Dhaka", "Osaka",
    "Karachi", "Chongqing", "Istanbul", "Buenos Aires", "Kolkata",
    "Kinshasa", "Lagos", "Manila", "Rio de Janeiro", "Guangzhou",
    "Los Angeles", "Moscow", "Paris", "Bangkok", "Jakarta",
    "London", "Lima", "New York", "Shenzhen", "Bangalore",
    "Ho Chi Minh City", "Hyderabad", "Bogota", "Tianjin", "Santiago",
    "Sydney", "Berlin", "Madrid", "Toronto", "Johannesburg",
    "Dubai", "Singapore", "Tehran", "Baghdad", "Riyadh",
    "Rome", "Cape Town", "Lagos", "Casablanca", "Barcelona",
    "Seoul", "Melbourne", "Copenhagen", "Zurich", "Kuala Lumpur"
]

# Travel time slots - extracted from both files (lines 108-115)
TRAVEL_TIME_SLOTS = [
    (5, 9, "Early Morning"),    # 5 AM - 8:55 AM
    (9, 12, "Late Morning"),    # 9 AM - 11:55 AM
    (12, 17, "Afternoon"),      # 12 PM - 4:55 PM
    (17, 21, "Evening"),        # 5 PM - 8:55 PM
    (21, 23, "Late Night"),     # 9 PM - 11:55 PM
    (23, 5, "Overnight")         # 12 AM - 4:55 AM
]

# Resolution statuses - extracted from both files (lines 116-120)
RESOLUTION_STATUSES = {
    "Resolved": 0.4,
    "Failed": 0.3,
    "Escalated": 0.3
}

# Service weights - extracted from both files (lines 759-764)
SERVICE_WEIGHTS = {
    'single': 0.30,    # 30% single service
    'double': 0.35,    # 35% two services
    'triple': 0.25,    # 25% three services
    'quadruple': 0.10  # 10% four services
}

# Core services list - extracted from both files (line 694)
CORE_SERVICES = ['hotel', 'restaurant', 'train', 'attraction', 'taxi', 'bus', 'hospital', 'flight']

# Logical combinations - extracted from both files (lines 695-758)
LOGICAL_COMBINATIONS = {
    'double': [
        # Hospital-related combinations
        ['hospital', 'taxi'],
        ['hospital', 'hotel'],
        # Flight-related combinations
        ['flight', 'taxi'],
        ['flight', 'hotel'],
        ['flight', 'train'],
        ['flight', 'bus'],
        ['flight', 'restaurant'],
        # Travel & Accommodation
        ['hotel', 'taxi'],
        ['hotel', 'train'],
        ['hotel', 'bus'],
        # Dining & Entertainment
        ['restaurant', 'taxi'],
        ['restaurant', 'attraction'],
        ['attraction', 'taxi'],
        # Transport Connections
        ['train', 'taxi'],
        ['bus', 'taxi'],
        ['train', 'bus'],
        # Common Pairings
        ['hotel', 'restaurant'],
    ],
    'triple': [
        # Hospital-related combinations
        ['hospital', 'taxi', 'hotel'],
        ['hospital', 'hotel', 'restaurant'],
        # Travel & Stay Combinations
        ['hotel', 'restaurant', 'taxi'],
        ['hotel', 'train', 'taxi'],
        ['hotel', 'bus', 'taxi'],
        # Tourism Combinations
        ['attraction', 'restaurant', 'taxi'],
        ['attraction', 'hotel', 'taxi'],
        ['attraction', 'train', 'taxi'],
        # Extended Travel Plans
        ['train', 'hotel', 'restaurant'],
        ['bus', 'hotel', 'restaurant'],
        ['train', 'restaurant', 'taxi'],
        # Flight-related triples
        ['flight', 'hotel', 'taxi'],
        ['flight', 'train', 'taxi'],
        ['flight', 'bus', 'taxi'],
        ['flight', 'restaurant', 'taxi'],
    ],
    'quadruple': [
        # Hospital-related combinations
        ['hospital', 'hotel', 'restaurant', 'taxi'],
        # Full Tourism Package
        ['hotel', 'restaurant', 'attraction', 'taxi'],
        ['train', 'hotel', 'restaurant', 'taxi'],
        ['bus', 'hotel', 'restaurant', 'taxi'],
        # Extended Tourism
        ['train', 'hotel', 'attraction', 'taxi'],
        ['bus', 'hotel', 'attraction', 'taxi'],
        ['flight', 'hotel', 'restaurant', 'taxi'],
        ['flight', 'train', 'hotel', 'taxi'],
        ['flight', 'bus', 'hotel', 'taxi'],
    ]
}