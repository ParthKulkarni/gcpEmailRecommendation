from flask import Flask, render_template, flash, redirect, url_for, session, request, logging, jsonify
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
import json
import model
import summarizer

app = Flask(__name__)

# Config MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'flaskapp'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# init MYSQL
mysql = MySQL(app)


# Index
@app.route('/')
def index():
    return render_template('home.html')


# About
@app.route('/about')
def about():
    return render_template('about.html')


# Single Thread
@app.route('/view_thread/<string:id>/')
def view_thread(id):
    # Create cursor
    cur = mysql.connection.cursor()

    # Get article
    result = cur.execute("SELECT * FROM mails1 WHERE thread_no = %s", [id])

    threads = cur.fetchall()

    if result > 0:
        return render_template('thread.html', threads=threads)
    else:
        msg = 'No mails Found'
        return render_template('thread.html', msg=msg)


# Register Form Class
class RegisterForm(Form):
    name = StringField('Name', [validators.Length(min=1, max=50)])
    username = StringField('Username', [validators.Length(min=4, max=25)])
    email = StringField('Email', [validators.Length(min=6, max=50)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords do not match')
    ])
    confirm = PasswordField('Confirm Password')


# User Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        password = sha256_crypt.encrypt(str(form.password.data))

        # Create cursor
        cur = mysql.connection.cursor()

        # Execute query
        cur.execute("INSERT INTO users1(name, email, username, password) VALUES(%s, %s, %s, %s)",
                    (name, email, username, password))

        # Commit to DB
        mysql.connection.commit()

        # Close connection
        cur.close()

        flash('You are now registered and can log in', 'success')

        return redirect(url_for('index'))
    return render_template('register.html', form=form)


# User login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get Form Fields
        username = request.form['username']
        password_candidate = request.form['password']

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        result = cur.execute(
            "SELECT * FROM users1 WHERE username = %s", [username])

        if result > 0:
            # Get stored hash
            data = cur.fetchone()
            password = data['password']

            # Compare Passwords
            if sha256_crypt.verify(password_candidate, password):
                # Passed
                session['logged_in'] = True
                session['username'] = username

                flash('You are now logged in', 'success')
                return redirect(url_for('dashboard'))
            else:
                error = 'Invalid login'
                return render_template('login.html', error=error)
            # Close connection
            cur.close()
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)

    return render_template('login.html')


# Check if user logged in
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap


# Logout
@app.route('/logout')
@is_logged_in
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('login'))


# Dashboard
@app.route('/dashboard', methods=['GET', 'POST'])
@is_logged_in
def dashboard():
    cur = mysql.connection.cursor()

    # # Get first 50 threads!!!!<figure out how to do it!!>
    result1 = cur.execute("SELECT * FROM threads1 LIMIT 20;")
    threads1 = cur.fetchall()
    # print(len(threads1))
    # print(type(threads1))

    query = 'SELECT * FROM threads1 WHERE id IN (SELECT DISTINCT thread_no FROM mails1 WHERE author = "' + \
        session['username'] + '");'
    result2 = cur.execute(query)
    threads2 = cur.fetchall()

    session['statement'] = 'These are all the email threads currently received!'
    if result1 > 0:
        # write query to get last 10 mails from user
        return render_template('dashboard.html', threads=threads1, replies=threads2)
    else:
        msg = 'No Articles Found'
        return render_template('dashboard.html', msg=msg)
    # Close connection
    cur.close()


# dashboardMailNum = 0


@app.route('/getNextMails', methods=['GET', 'POST'])
def getNextMails():
    start = request.args['key']
    cur = mysql.connection.cursor()
    query = "SELECT * FROM threads1 LIMIT " + str(start) + ", 20"
    result1 = cur.execute(query)
    threads = cur.fetchall()
    cur.close()
    return json.dumps(threads, default=str)


def max(i, j):
    if(i >= j):
        return i
    else:
        return j


@app.route('/getPreviousMails', methods=['GET', 'POST'])
def getPreviousMails():
    # do stuff to get previous 50 threads.
    start = int(request.args['key'])
    start = max(0, start)
    cur = mysql.connection.cursor()
    query = "SELECT * FROM threads1 LIMIT " + str(start) + ", 20"
    result1 = cur.execute(query)
    threads = cur.fetchall()
    cur.close()

    return json.dumps(threads, default=str)


def list_to_string(l):
    s = ''
    for item in l:
        s = s + str(item) + ","
    s = s[:-1]
    return s


@app.route('/recommend', methods=['GET', 'POST'])
def recommmend():
    if session.get('logged_in', False):
        name = session['username']
        print(name)
        ob = model.Recom()
        threadList = ob.getThreads(name)
        query = ''
        if len(threadList) == 0:
            query = "SELECT * FROM threads1 LIMIT 10;"
        else:
            print(list_to_string(threadList))
            query = """SELECT * FROM  threads1 WHERE id in (""" + list_to_string(threadList) + """);"""
            print(query)
            # query = """SELECT * FROM  threads1 where id in (283,2091,3046,3169,3637);"""
        cur = mysql.connection.cursor()
        result1 = cur.execute(query)
        threads = cur.fetchall()
        cur.close()
        if result1 > 0:
            print(len(threads))
            return render_template('recom.html', threads=threads)
        else:
            msg = 'No Articles Found'
            return render_template('recom.html', msg=msg)
    else :
        return redirect(url_for('login'))



@app.route('/getSummarizeFeed', methods=['GET', 'POST'])
def getSummarizeFeed():
    threadId = request.args['key']
    cur = mysql.connection.cursor()
    query = "SELECT * FROM mails1 WHERE thread_no = " + threadId + ";"
    result1 = cur.execute(query)
    mails = cur.fetchall()
    # print("query : " + query)
    cur.close()
    subject = ""
    sents = []
    subject = mails[0]['subject']
    for mail in mails:
        sents.append(mail['content'])

    # print(sents)

    ob = summarizer.Summarizer(subject, sents)
    data = ob.generate_summary()
    return data

    # ob = summarizer.Summarizer()
    # print(mails[0])


if __name__ == '__main__':
    app.secret_key = 'secret123'
    app.run(debug=True)


# SELECT * FROM threads WHERE id IN (SELECT DISTINCT thread_no FROM mails WHERE author = 'Dimitri John Ledkov');
