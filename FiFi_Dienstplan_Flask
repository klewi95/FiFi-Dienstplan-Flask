from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
import pandas as pd
import pulp
from datetime import datetime, timedelta
import holidays
import json
import os
import csv
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Ersetzen Sie durch einen sicheren Schlüssel

# Datei für Mitarbeiterdaten
EMPLOYEE_FILE = 'employees.json'

# Funktion zum Laden der Mitarbeiterdaten
def load_employees():
    if not os.path.exists(EMPLOYEE_FILE):
        return {}
    with open(EMPLOYEE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

# Funktion zum Speichern der Mitarbeiterdaten
def save_employees(employees):
    with open(EMPLOYEE_FILE, 'w', encoding='utf-8') as f:
        json.dump(employees, f, ensure_ascii=False, indent=4)

# Laden der Mitarbeiterdaten beim Start der Anwendung
employees = load_employees()

# Definition der Schichten und deren Dauer sowie Startzeiten
shifts_weekday = {
    'Frühschicht': {'duration': 8, 'start': 6.75},    # 6:45 Uhr
    'Spätschicht': {'duration': 8, 'start': 14.75}    # 14:45 Uhr
}

shifts_weekend = {
    'Frühschicht': {'duration': 5, 'start': 9.25},    # 9:15 Uhr
    'Spätschicht': {'duration': 6, 'start': 14.25}    # 14:15 Uhr
}

# Betriebsratvorschriften und Arbeitszeitgesetz
max_consecutive_days = 6  # Maximal 6 aufeinanderfolgende Arbeitstage
max_daily_hours = 8       # Maximal 8 Stunden pro Tag
min_rest_time = 11        # Mindestruhezeit zwischen Schichten in Stunden

# Parameter für Fairness und Soft Constraints
allowed_shift_deviation = 2  # Erlaubte Abweichung von der durchschnittlichen Schichtanzahl
penalty_per_day = 100        # Strafwert für jeden Tag über dem Soft-Limit

# Zeitraum für den Dienstplan (standardmäßig aktueller Monat)
start_date = datetime.now().replace(day=1)
end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
dates = pd.date_range(start_date, end_date)

# Feiertage automatisch berechnen
year = start_date.year
deutschland_feiertage = holidays.Germany(years=year, prov='NW')  # 'NW' für Nordrhein-Westfalen
feiertage = set([date for date in dates if date in deutschland_feiertage])

# Hilfsfunktionen zur Bestimmung von Wochentagen und Schichtinformationen
def is_weekend_or_holiday(date):
    return date.weekday() >= 5 or date in feiertage

def get_shift_duration(shift, date):
    is_weekend = is_weekend_or_holiday(date)
    if is_weekend:
        return shifts_weekend[shift]['duration']
    else:
        return shifts_weekday[shift]['duration']

def get_shift_start(shift, date):
    is_weekend = is_weekend_or_holiday(date)
    if is_weekend:
        return shifts_weekend[shift]['start']
    else:
        return shifts_weekday[shift]['start']

def get_actual_working_time(shift, date):
    duration = get_shift_duration(shift, date)
    if duration > 6:
        return duration - 1  # Abzug der 1-stündigen Pause
    else:
        return duration  # Keine Pause erforderlich

def get_preference_score(employee, date, shift):
    preferences = employees[employee].get('preferences', {})
    date_str = date.strftime('%Y-%m-%d')
    day_name = date.strftime('%A')

    # Prüfe auf spezifische Datumsvorgabe
    date_pref = preferences.get(date_str)
    if date_pref is not None:
        return date_pref.get(shift, 0)

    # Prüfe auf Wochentagspräferenz
    day_pref = preferences.get(day_name)
    if day_pref is not None:
        return day_pref.get(shift, 0)

    # Standardpräferenz
    return 0

# Funktion zur Generierung des Dienstplans
def generate_schedule():
    global employees
    if not employees:
        return None, "Keine Mitarbeiterdaten vorhanden."

    # Initialisieren des Optimierungsproblems
    prob = pulp.LpProblem("Dienstplan", pulp.LpMaximize)

    # Entscheidungsvariablen: Zuordnung von Mitarbeitern zu Schichten
    assignments = {}
    for employee in employees:
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            is_weekend = is_weekend_or_holiday(date)
            available_shifts = shifts_weekend.keys() if is_weekend else shifts_weekday.keys()
            for shift in available_shifts:
                var = pulp.LpVariable(f"{employee}_{date_str}_{shift}", cat='Binary')
                assignments[(employee, date_str, shift)] = var

    # Berechnung der durchschnittlichen Schichten pro Mitarbeiter
    total_shifts = len(dates) * 2  # Pro Tag 2 Schichten
    num_employees = len(employees)
    total_max_hours = sum([employees[e]['max_weekly_hours'] for e in employees])
    average_shifts_per_employee = {e: (employees[e]['max_weekly_hours'] / total_max_hours) * total_shifts for e in employees}

    # Gewichtungen für die Zielfunktion
    preference_weight = 10  # Gewichtung der Präferenzen
    penalty_weight = 1      # Gewichtung der Strafwerte

    # Berechnung der Präferenzpunkte
    preference_score = pulp.lpSum([
        assignments.get((e, d.strftime('%Y-%m-%d'), s), 0) * get_preference_score(e, d, s)
        for e in employees
        for d in dates
        for s in shifts_weekday.keys()
        if (e, d.strftime('%Y-%m-%d'), s) in assignments
    ])

    # Strafwerte für Soft Constraints (maximale aufeinanderfolgende Arbeitstage)
    penalty_terms = []
    consecutive_days_vars = {}
    for e in employees:
        for idx in range(len(dates) - max_consecutive_days):
            var = pulp.LpVariable(f"ConsecutiveDays_{e}_{idx}", lowBound=0, cat='Integer')
            consecutive_days_vars[(e, idx)] = var
            # Berechnung der aufeinanderfolgenden Arbeitstage
            total_consecutive = pulp.lpSum([
                pulp.lpSum([assignments.get((e, dates[idx + j].strftime('%Y-%m-%d'), shift), 0) for shift in shifts_weekday.keys()])
                for j in range(max_consecutive_days + 1)
            ])
            prob += var >= total_consecutive - max_consecutive_days, f"SoftMaxConsecutiveDays_{e}_{idx}"
            # Hinzufügen zum Strafwert
            penalty_terms.append(var * penalty_per_day)

    # Zielfunktion: Maximierung der Präferenzen minus Strafwerte
    prob += preference_weight * preference_score - penalty_weight * pulp.lpSum(penalty_terms), "ObjectiveFunction"

    # Nebenbedingungen

    # 1. Personalbesetzung pro Schicht
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        is_weekend = is_weekend_or_holiday(date)
        available_shifts = shifts_weekend.keys() if is_weekend else shifts_weekday.keys()
        for shift in available_shifts:
            total_staff = pulp.lpSum([assignments.get((e, date_str, shift), 0) for e in employees])
            # Mindestpersonal
            prob += total_staff >= 2, f"MinStaff_{date_str}_{shift}"
            # Maximalpersonal
            prob += total_staff <= 3, f"MaxStaff_{date_str}_{shift}"

    # 2. Wöchentliche Arbeitszeit pro Mitarbeiter
    weeks = {}
    for date in dates:
        week_num = date.isocalendar()[1]
        if week_num not in weeks:
            weeks[week_num] = []
        weeks[week_num].append(date.strftime('%Y-%m-%d'))

    for e in employees:
        for week_num, week_dates in weeks.items():
            total_weekly_hours = pulp.lpSum([
                (assignments.get((e, d, shift), 0) * get_actual_working_time(shift, pd.to_datetime(d)))
                for d in week_dates
                for shift in shifts_weekday.keys()
            ])
            # Maximale wöchentliche Arbeitszeit
            prob += total_weekly_hours <= employees[e]['max_weekly_hours'], f"MaxWeeklyHours_{e}_Week_{week_num}"
            # Minimale wöchentliche Arbeitszeit
            prob += total_weekly_hours >= employees[e]['min_weekly_hours'], f"MinWeeklyHours_{e}_Week_{week_num}"

    # 3. Maximale tägliche Arbeitszeit
    for e in employees:
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            daily_hours = pulp.lpSum([
                (assignments.get((e, date_str, shift), 0) * get_actual_working_time(shift, date))
                for shift in shifts_weekday.keys()
            ])
            prob += daily_hours <= max_daily_hours, f"MaxDailyHours_{e}_{date_str}"

    # 4. Maximal eine Schicht pro Tag pro Mitarbeiter
    for e in employees:
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            prob += pulp.lpSum([
                assignments.get((e, date_str, shift), 0)
                for shift in shifts_weekday.keys()
            ]) <= 1, f"MaxOneShiftPerDay_{e}_{date_str}"

    # 5. Berücksichtigung der Verfügbarkeiten und Einschränkungen
    for e in employees:
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            day_name = date.strftime('%A')
            available_shifts = employees[e]['availability'].get(day_name, [])
            # Einschränkungen für bestimmte Tage
            restricted_shifts = employees[e].get('restrictions', {}).get(date_str, [])
            for shift in shifts_weekday.keys():
                if shift not in available_shifts or shift in restricted_shifts:
                    if (e, date_str, shift) in assignments:
                        prob += assignments[(e, date_str, shift)] == 0, f"Restriction_{e}_{date_str}_{shift}"

    # 6. Maximal erlaubte aufeinanderfolgende Arbeitstage (harte Grenze)
    for e in employees:
        for idx in range(len(dates) - max_consecutive_days):
            total_consecutive = pulp.lpSum([
                pulp.lpSum([assignments.get((e, dates[idx + j].strftime('%Y-%m-%d'), shift), 0) for shift in shifts_weekday.keys()])
                for j in range(max_consecutive_days + 1)
            ])
            prob += total_consecutive <= max_consecutive_days, f"MaxConsecutiveDays_{e}_{idx}"

    # 7. Mindestruhezeit zwischen Schichten
    for e in employees:
        for idx in range(len(dates) - 1):
            current_date = dates[idx]
            next_date = dates[idx + 1]
            current_date_str = current_date.strftime('%Y-%m-%d')
            next_date_str = next_date.strftime('%Y-%m-%d')
            for current_shift in shifts_weekday.keys():
                for next_shift in shifts_weekday.keys():
                    if (e, current_date_str, current_shift) in assignments and (e, next_date_str, next_shift) in assignments:
                        # Endzeit der aktuellen Schicht
                        end_time_current = get_shift_start(current_shift, current_date) + get_shift_duration(current_shift, current_date)
                        # Startzeit der nächsten Schicht
                        start_time_next = get_shift_start(next_shift, next_date)
                        # Berechnung der Ruhezeit
                        rest_time = (start_time_next + (24 if start_time_next <= end_time_current else 0)) - end_time_current
                        if rest_time < min_rest_time:
                            prob += assignments[(e, current_date_str, current_shift)] + assignments[(e, next_date_str, next_shift)] <= 1, f"MinRestTime_{e}_{current_date_str}_{current_shift}_{next_date_str}_{next_shift}"

    # 8. Fairness in der Schichtverteilung
    for e in employees:
        total_shifts_assigned = pulp.lpSum([
            assignments.get((e, d.strftime('%Y-%m-%d'), shift), 0)
            for d in dates
            for shift in shifts_weekday.keys()
        ])
        # Fairnessbedingungen
        prob += total_shifts_assigned >= average_shifts_per_employee[e] - allowed_shift_deviation, f"FairnessMin_{e}"
        prob += total_shifts_assigned <= average_shifts_per_employee[e] + allowed_shift_deviation, f"FairnessMax_{e}"

    # 9. Fairness bei Wochenend- und Feiertagsschichten
    def calculate_weekend_holiday_shifts(employee):
        return pulp.lpSum([
            assignments.get((employee, d.strftime('%Y-%m-%d'), s), 0)
            for d in dates
            for s in shifts_weekday.keys()
            if is_weekend_or_holiday(d)
        ])

    total_weekend_holiday_shifts = pulp.lpSum([
        calculate_weekend_holiday_shifts(e)
        for e in employees
    ])

    avg_weekend_holiday_shifts = total_weekend_holiday_shifts / num_employees if num_employees > 0 else 0

    for e in employees:
        total_weekend_shifts = calculate_weekend_holiday_shifts(e)
        prob += total_weekend_shifts <= avg_weekend_holiday_shifts + 1, f"FairWeekendHoliday_{e}_Max"
        prob += total_weekend_shifts >= avg_weekend_holiday_shifts - 1, f"FairWeekendHoliday_{e}_Min"

    # 10. Rolling Average von 48 Stunden pro Woche über 4 Wochen
    for e in employees:
        for idx in range(len(dates)):
            if idx >= 27:  # Betrachtung der letzten 4 Wochen (28 Tage)
                start_idx = idx - 27
                period_dates = dates[start_idx:idx+1]
                total_hours = pulp.lpSum([
                    (assignments.get((e, d.strftime('%Y-%m-%d'), shift), 0) * get_actual_working_time(shift, d))
                    for d in period_dates
                    for shift in shifts_weekday.keys()
                ])
                prob += total_hours <= 48 * 4, f"RollingAvg48h_{e}_{dates[idx].strftime('%Y-%m-%d')}"

    # Lösen des Optimierungsproblems
    prob.solve()

    # Überprüfen, ob eine optimale Lösung gefunden wurde
    if pulp.LpStatus[prob.status] != 'Optimal':
        return None, f"Keine optimale Lösung gefunden. Status: {pulp.LpStatus[prob.status]}"

    # Erstellen des Dienstplans basierend auf der Lösung
    dienstplan = {e: [] for e in employees}

    for e in employees:
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            day_name = date.strftime('%A')
            for shift in shifts_weekday.keys():
                if (e, date_str, shift) in assignments and pulp.value(assignments[(e, date_str, shift)]) == 1:
                    start_time = get_shift_start(shift, date)
                    end_time = start_time + get_shift_duration(shift, date)
                    # Umrechnung der Zeiten in Stunden und Minuten
                    start_hour = int(start_time)
                    start_min = int((start_time - start_hour) * 60)
                    end_hour = int(end_time)
                    end_min = int((end_time - end_hour) * 60)
                    # Anpassung der Endzeit bei Überlauf über 24 Stunden
                    if end_hour >= 24:
                        end_hour -= 24
                    start_str = f"{start_hour:02d}:{start_min:02d}"
                    end_str = f"{end_hour:02d}:{end_min:02d}"
                    # Berechnung der tatsächlichen Arbeitszeit
                    duration = get_shift_duration(shift, date)
                    if duration > 6:
                        pause = True
                        arbeitszeit = duration - 1  # Abzug der Pause
                    else:
                        pause = False
                        arbeitszeit = duration
                    # Hinzufügen zur Dienstplanliste
                    dienstplan[e].append({
                        'Datum': date_str,
                        'Wochentag': day_name,
                        'Schicht': shift,
                        'Startzeit': start_str,
                        'Endzeit': end_str,
                        'Arbeitszeit (Std.)': arbeitszeit,
                        'Pause (1 Std.)': 'Ja' if pause else 'Nein'
                    })

    return dienstplan, "Optimal"

# Route für die Startseite
@app.route('/')
def index():
    return render_template('index.html')

# Route für die Mitarbeiterverwaltung
@app.route('/manage_employees')
def manage_employees():
    employees = load_employees()
    return render_template('manage_employees.html', employees=employees)

# Route zum Hinzufügen oder Bearbeiten eines Mitarbeiters
@app.route('/add_edit_employee', methods=['GET', 'POST'])
def add_edit_employee():
    employees = load_employees()
    if request.method == 'POST':
        name = request.form.get('name')
        max_hours = int(request.form.get('max_hours'))
        min_hours = int(request.form.get('min_hours'))
        availability = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for day in days:
            shifts = request.form.getlist(day)
            availability[day] = shifts
        employees[name] = {
            "max_weekly_hours": max_hours,
            "min_weekly_hours": min_hours,
            "availability": availability,
            "restrictions": {},
            "preferences": {}
        }
        save_employees(employees)
        flash("Mitarbeiterdaten gespeichert.", "success")
        return redirect(url_for('manage_employees'))
    else:
        name = request.args.get('name')
        if name:
            employee = employees.get(name)
        else:
            employee = None
        return render_template('add_edit_employee.html', employee=employee)

# Route zum Löschen eines Mitarbeiters
@app.route('/delete_employee/<name>')
def delete_employee(name):
    employees = load_employees()
    if name in employees:
        del employees[name]
        save_employees(employees)
        flash("Mitarbeiter gelöscht.", "success")
    return redirect(url_for('manage_employees'))

# Route für die Dienstplanerstellung
@app.route('/generate_schedule', methods=['GET', 'POST'])
def generate_schedule_route():
    if request.method == 'POST':
        # Zeitraum aus dem Formular
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')
        if start_date_str and end_date_str:
            global start_date, end_date, dates, feiertage
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            if start_date > end_date:
                flash("Das Startdatum darf nicht nach dem Enddatum liegen.", "danger")
                return redirect(url_for('generate_schedule_route'))
            dates = pd.date_range(start_date, end_date)
            year = start_date.year
            deutschland_feiertage = holidays.Germany(years=year, prov='NW')
            feiertage = set([date for date in dates if date in deutschland_feiertage])
        dienstplan, status = generate_schedule()
        if dienstplan:
            # Speichern des Dienstplans in der Session
            session['dienstplan'] = dienstplan
            flash("Dienstplan erfolgreich erstellt.", "success")
            return redirect(url_for('view_schedule'))
        else:
            flash(f"Fehler bei der Dienstplanerstellung: {status}", "danger")
    return render_template('generate_schedule.html')

# Route zur Anzeige des Dienstplans
@app.route('/view_schedule')
def view_schedule():
    dienstplan = session.get('dienstplan', None)
    if dienstplan:
        return render_template('view_schedule.html', dienstplan=dienstplan)
    else:
        flash("Kein Dienstplan gefunden. Bitte erstellen Sie zuerst einen Dienstplan.", "info")
        return redirect(url_for('generate_schedule_route'))

# Route zum Herunterladen des Dienstplans als CSV
@app.route('/download_schedule')
def download_schedule():
    dienstplan = session.get('dienstplan', None)
    if dienstplan:
        output = io.StringIO()
        fieldnames = ['Mitarbeiter', 'Datum', 'Wochentag', 'Schicht', 'Startzeit', 'Endzeit', 'Arbeitszeit (Std.)', 'Pause (1 Std.)']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for e, shifts in dienstplan.items():
            for s in shifts:
                writer.writerow({
                    'Mitarbeiter': e,
                    'Datum': s['Datum'],
                    'Wochentag': s['Wochentag'],
                    'Schicht': s['Schicht'],
                    'Startzeit': s['Startzeit'],
                    'Endzeit': s['Endzeit'],
                    'Arbeitszeit (Std.)': s['Arbeitszeit (Std.)'],
                    'Pause (1 Std.)': s['Pause (1 Std.)']
                })
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            attachment_filename='dienstplan.csv')
    else:
        flash("Kein Dienstplan zum Herunterladen gefunden.", "info")
        return redirect(url_for('generate_schedule_route'))

if __name__ == '__main__':
    app.run(debug=False)
