import pyautogui
import time
from datetime import datetime
import pyscreeze

url = "https://www.nseindia.com/market-data/live-equity-market?symbol=NIFTY%2050"

print ("step 1 open chrome and got to nseindia.com")
time.sleep(2)
pyautogui.hotkey('win', 'r',interval=0.1)
time.sleep(2)
pyautogui.typewrite('chrome\n', interval=0.1)
time.sleep(2)
pyautogui.press('enter', interval=0.1)
time.sleep(2)

pyautogui.press('tab', interval=0.1)

time.sleep(2)

pyautogui.press('enter', interval=0.1)
time.sleep(2)

#open new tan
pyautogui.hotkey('ctrl', 'l',interval=0.1)


pyautogui.typewrite(f'{url}\n', interval=0.1)

print ("step 2 wait for 10 seconds to load the page")
time.sleep(10)

print ("step 3 copy nifty 50 data")
pyautogui.hotkey('ctrl', 'a')
pyautogui.hotkey('ctrl', 'c')

print ("step 4 open excel")

time.sleep(2)
pyautogui.hotkey('win', 'r')
pyautogui.typewrite('excel\n', interval=0.1)
time.sleep(10)
pyautogui.press('enter')
time.sleep(10)



pyautogui.press('enter')


print ("step 5 wait for 10 seconds to load the excel")
time.sleep(10)

# Make sure Excel is active and start at cell A1
pyautogui.hotkey("ctrl", "home")

# Column A: date and time
pyautogui.write(time.strftime("%Y-%m-%d %H:%M:%S"), interval=0.05)
pyautogui.press("tab")

# Column B: paste the data copied earlier
pyautogui.hotkey("ctrl", "v")
pyautogui.press("tab")

# Column C: your comment
pyautogui.write("NIFTY 50 data collected", interval=0.05)


print("Step 6: Save the workbook to Desktop")
pyautogui.press("f12")
time.sleep(3)

file_path = "C:\\users\\Lenovo\\OneDrive\\Desktop"
pyautogui.hotkey('ctrl', 'l')
time.sleep(2)
pyautogui.typewrite(file_path, interval=0.1)
pyautogui.press("enter")
time.sleep(2)
pyautogui.press("tab", presses=5, interval=0.2)
pyautogui.hotkey('alt', 'n')
filename = datetime.now().strftime("%Y-%m-%d") + ".xlsx"
pyautogui.write(filename, interval=0.1)
pyautogui.press("enter")
pyautogui.screenshot("C:\\users\\Lenovo\\OneDrive\\Desktop\\screenshot.png")

print("Finished")
