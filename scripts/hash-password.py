#!/usr/bin/env python3
"""Create a password hash using an interactive prompt; never print plaintext."""
import getpass
import hashlib
import secrets
password=getpass.getpass('LINEAGE password (12+ characters): ')
if len(password)<12:raise SystemExit('Use at least 12 characters.')
if password!=getpass.getpass('Repeat password: '):raise SystemExit('Passwords differ.')
salt=secrets.token_hex(16)
print(salt+':'+hashlib.pbkdf2_hmac('sha256',password.encode(),salt.encode(),600000).hex())
