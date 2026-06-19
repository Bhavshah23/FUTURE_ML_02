from src.predict import predict_ticket

print("=" * 50)
print("SUPPORT TICKET INTELLIGENCE SYSTEM")
print("=" * 50)

ticket = input("\nEnter Ticket Description:\n")

category, priority = predict_ticket(ticket)

print("\nResults")
print("-" * 30)

print("Category :", category)
print("Priority :", priority)