# IT_Ticket_Intake
ML Solution for Ticket Intake

Ownership & Maintenance:
This repository is owned by Cantel IT and maintained by the Advanced Analytics organization.

Overview:
This repository will serve as the production and sandbox instance for ML development pertaining to the IT Ticket Intake process.
The ML solution is intended to reduce cost and latency for the current process within IT ticket intake via ServiceNow. 
The model evaluates accuracy, precision and recall to make a decision concerning whether or not to publish predictions.
The scripts utilize ServiceNow API (GET / PUT) to extract and publish data.

Current Process:
The IT department relies on a 3rd party vendor to examine new tickets that flow into ServiceNow. The vendor is responsible for 
filling-out specific fields to correctly categorize the ticket, including "Business Service", "Assignment Group", and "Assigned-To".
Every interaction with a ticket costs the organization $13 dollars; anecdotal evidence suggests poor accuracy and latency.
On average, 1600 tickets flow through this process every month.

Improved Process:
Using a Machine Learning model to make predictions, the Advanced Analytics organization can improve accuracy and latency while drastically
reducing costs for IT ticket interactions. We used a training dataset of over 25,000 archived tickets to build a Supervised, 
Classification algorithm. The algorithm predicts the fields "Portfolio" and "Assignment Group" to bypass 3rd party vendor interaction 
and automate the flow of new tickets in ServiceNow to the correct Portfolio owner and/or Assignment Group. The model will pull data from
ServiceNow upon new ticket creation, run the text from the ticket through the ML model, determine whether or not the model detects
significance, evaluate probability, and update the appropriate fields directly in the ServiceNow platform.
