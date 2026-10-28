import sys, urllib.request
H = {
 "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
 "Referer":"https://radio.or.ke/","Origin":"https://radio.or.ke",
 "Accept":"application/json, text/plain, */*",
}
for u in sys.argv[1:]:
    print("===", u)
    try:
        print(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=25).read().decode("utf8","replace")[:1200])
    except Exception as e:
        print("FAIL", type(e).__name__, e)
