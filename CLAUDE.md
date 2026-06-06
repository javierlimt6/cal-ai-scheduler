# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is a coding challenge: build an interactive chatbot (in **Python** — required) that lets a user manage their cal.com account through plain conversation. The target user is a busy founder who wants to type things like "book a 30-min intro with a candidate Thursday afternoon" or "move my 3pm to later today" — no forms or menus.

As of now the repository contains only the README; the implementation has not been started.

## Requirements (from README.md)

The chatbot must interact with the cal.com REST API and support, through conversation:

1. **Book** a new event — the assistant gathers whatever details it needs, then creates the event
2. **View** scheduled events
3. **Cancel** an event
4. **Reschedule** an existing booking

Any LLM provider may be used. An interactive web UI is a plus, not a requirement. Guiding principle from the README: "Build the experience you'd want this user to have."

## cal.com API (v2)

A cal.com account and API key are prerequisites. Key documentation:

- Authentication / API key: https://cal.com/docs/enterprise-features/api/authentication
- Bookings API: https://cal.com/docs/api-reference/v2/bookings/get-all-bookings
- Slots API (availability): https://cal.com/docs/api-reference/v2/slots/find-out-when-is-an-event-type-ready-to-be-booked

The booking flow generally requires checking available slots (Slots API) before creating a booking, and bookings are tied to an event type.
