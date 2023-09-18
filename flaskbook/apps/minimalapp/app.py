from flask import Flask, render_template, url_for, current_app, g, render_template, url_for, request, redirect, flash
from email_validator import validate_email, EmailNotValidError
import logging
import os
from flask_mail import Mail, Message
from flask_debugtoolbar import DebugToolbarExtension\
import logging

app = Flask(__name__)
app.config["config_key"] = "2AZSMss3p5QPbcY2hBs"
app.logger.setLevel(logging.DEBUG)
app.config["DEBUG_TB_INTERCEPT_REDITRCTS"] = False
app.config["MAIL_SERVER"] = os.environ.get("MAIL_SERVER")
app.config["MAIL_PORT"] = os.environ.get("MAIL_PORT")
app.config["MAIL_USE_TLS"] = os.environ.get("MAIL_USE_TLS")
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_DEFAULT_SENDER")
mail = Mail(app)
toolbar = DebugToolbarExtension(app)

@app.route('/')
def index():
    return "hellow, flaskbook!"


@app.route("/hello/<name>", methods=["GET", "POST"], endpoint="hello-endpoint")
def hello(name):
    return f"Hello, {name}!"

@app.route("/name/<name>")
def show_name(name):
    return render_template("index.html", name=name)

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/contact/complete", methods=["GET", "POST"])
def contact_complete():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        description = request.form["description"]

        is_valid = True

        if not username:
            flash("사용자명은 필수입니다")
            is_valid = False

        if not email:
            flash("메일 주소는 필수입니다")
            is_valid = False
        try:
            validate_email(email)
        except EmailNotValidError:
            flash("메일 주소의 형식으로 입력해 주세요")
            is_valid = False

        if not description:
            flash("문의 내용은 필수입니다")
            is_valid = False

        if not is_valid:
            return redirect(url_for("contact"))
        
        # flash("문의해 주셔서 감사합니다.")
        send_email(
            email,
            "문의 감사합니다.",
            "contact_email",
            username=username,
            description=description,
        )

        return redirect(url_for("contact_complete"))

    return render_template("contact_complete.html")
