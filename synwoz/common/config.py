from typing import List, Dict, Tuple

# Shared constants copied from existing scripts to avoid divergence

TRAVEL_TIME_SLOTS: List[Tuple[int, int, str]] = [
    (5, 9, "Early Morning"),
    (9, 12, "Late Morning"),
    (12, 17, "Afternoon"),
    (17, 21, "Evening"),
    (21, 23, "Late Night"),
    (23, 5, "Overnight"),
]

RESOLUTION_STATUSES: Dict[str, float] = {
    "Resolved": 0.4,
    "Failed": 0.3,
    "Escalated": 0.3,
}

USER_EMOTION_LIST: List[str] = [
    "Frustrated", "Angry", "Confused", "Worried", "Disappointed",
    "Happy", "Anxious", "Impatient", "Skeptical", "Desperate",
    "Overwhelmed", "Hopeful", "Satisfied", "Stressed", "Suspicious",
    "Tired", "Excited", "Indifferent", "Grateful", "Demanding",
]

ASSISTANT_EMOTION_LIST: List[str] = [
    "Professional", "Informative", "Reassuring", "Diplomatic", "Patient",
    "Efficient", "Accommodating", "Solution-focused", "Methodical", "Proactive",
    "Analytical", "Composed", "Detail-oriented", "Responsive", "Thorough",
    "Systematic", "Precise", "Objective", "Resourceful", "Knowledgeable",
]

SCENARIO_CATEGORIES: Dict[str, List[str]] = {
    "general": [
        "account_management", "cancellation_general", "complaint",
        "refund_request_general", "payment_issues", "general_inquiry",
        "feedback", "technical_support", "lost_and_found_general",
    ],
    "restaurant": [
        "dining_reservation", "dietary_requirements", "table_modification",
        "special_occasion", "menu_inquiry", "order_status",
        "reservation_cancellation",
    ],
    "hotel": [
        "room_reservation", "check_in_out", "amenity_inquiry", "room_service",
        "booking_modification", "housekeeping_request", "reservation_cancellation",
    ],
    "train": [
        "journey_planning", "schedule_inquiry", "ticket_booking",
        "platform_information", "ticket_change", "ticket_cancellation",
        "seat_selection",
    ],
    "attraction": [
        "ticket_availability", "opening_hours", "guided_tour",
        "venue_information", "group_booking", "ticket_cancellation",
        "accessibility_inquiry",
    ],
    "taxi": [
        "ride_booking", "pickup_location", "fare_inquiry", "driver_tracking",
        "ride_cancellation", "ride_feedback", "service_type_inquiry",
    ],
    "hospital": [
        "appointment_booking", "department_inquiry", "medical_information",
        "emergency_services", "appointment_cancellation", "insurance_inquiry",
        "medical_record_request",
    ],
    "bus": [
        "route_information", "schedule_inquiry", "ticket_booking",
        "stop_location", "ticket_change", "ticket_cancellation",
        "seat_selection",
    ],
    "flight": [
        "flight_booking", "cancellation_flight", "ticket_change",
        "baggage_inquiry", "check_in", "seat_selection", "flight_status",
        "upgrade_request", "refund_request_flight", "lounge_access",
        "boarding_pass_issue", "special_meals", "pet_transportation",
    ],
}

PREDEFINED_REGIONS: List[str] = [
    "Tokyo", "Delhi", "Shanghai", "Sao Paulo", "Mumbai", "Beijing", "Cairo",
    "Mexico City", "Dhaka", "Osaka", "Karachi", "Chongqing", "Istanbul",
    "Buenos Aires", "Kolkata", "Kinshasa", "Lagos", "Manila", "Rio de Janeiro",
    "Guangzhou", "Los Angeles", "Moscow", "Paris", "Bangkok", "Jakarta",
    "London", "Lima", "New York", "Shenzhen", "Bangalore", "Ho Chi Minh City",
    "Hyderabad", "Bogota", "Tianjin", "Santiago", "Sydney", "Berlin", "Madrid",
    "Toronto", "Johannesburg", "Dubai", "Singapore", "Tehran", "Baghdad",
    "Riyadh", "Rome", "Cape Town", "Casablanca", "Barcelona", "Seoul",
    "Melbourne", "Copenhagen", "Zurich", "Kuala Lumpur",
]


