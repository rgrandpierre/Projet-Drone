from pyparrot.Bebop import Bebop
import time
import threading
 
def emergency_landing(bebop):
    print("Atterrissage d'urgence déclenché !")
    bebop.emergency_land()

def monitor_emergency(bebop):
    while True:
        user_input = input("Tapez 'emergency' pour atterrissage d'urgence : ")
        if user_input.lower() == 'emergency':
            emergency_landing(bebop)
            break

bebop = Bebop(drone_type="Bebop2", ip_address="192.168.42.1")

print("Connexion")
success = bebop.connect(10)
print(success)

if success:
    # Démarrer un thread pour check l'atterrissage d'urgence
    threading.Thread(target=monitor_emergency, args=(bebop,), daemon=True).start()
    bebop.set_max_altitude(5)
    bebop.set_max_distance(10)
    bebop.set_max_tilt(5)
    bebop.set_max_vertical_speed(1)
    bebop.enable_geofence(1)
    bebop.start_video_stream()

    bebop.smart_sleep(2)
    senariot=1
    print("ok1")
    bebop.ask_for_state_update()
    print("ok")
    if (senariot==1):
        print("Sénariot 1")
        bebop.safe_takeoff(20)
        bebop.fly_direct(roll=0, pitch=10, yaw=0, vertical_movement=20, duration=1)
        bebop.fly_direct(roll=0, pitch=0, yaw=90, vertical_movement=20, duration=7)
        bebop.fly_direct(roll=0, pitch=0, yaw=90, vertical_movement=20, duration=7)
        bebop.smart_sleep(5)
        bebop.fly_direct(roll=0, pitch=0, yaw=-90, vertical_movement=20, duration=7)
        bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-20, duration=1)
    
    if (senariot==2):
        print("Sénariot 2")
        bebop.safe_takeoff(20)
        bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=20, duration=1)
        bebop.fly_direct(roll=0, pitch=0, yaw=90, vertical_movement=20, duration=7)
        bebop.smart_sleep(5)
        bebop.fly_direct(roll=0, pitch=0, yaw=-90, vertical_movement=20, duration=7)
        bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-20, duration=1)
    

    if (senariot==3):
        print("Sénariot 3")
        bebop.safe_takeoff(20)
        bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=20, duration=1)
        bebop.fly_direct(roll=0, pitch=20, yaw=90, vertical_movement=20, duration=7)
        bebop.smart_sleep(5)
        bebop.fly_direct(roll=0, pitch=-20, yaw=-90, vertical_movement=20, duration=7)
        bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-20, duration=1)
    

    bebop.safe_land(10)

    print("Fin")
    bebop.stop_video_stream()
    bebop.smart_sleep(5)
    #print(bebop.sensors.battery)
    ask_for_state_update()
    bebop.disconnect()
    

    
else:
    print("Échec de la connexion au drone.")
