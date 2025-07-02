import time
import matplotlib.pyplot as plt
from utils import MQTTMultiSourceManager, camera_handler
from utils import TrackMapper, enhanced_on_new_track, enhanced_on_track_lost, enhanced_on_track_update



if __name__ == "__main__":
    MAP_IMAGE_PATH = "rooms_database/room0/2Dmap.png"
    COORDINATE_BOUNDS = {
        'x_min': 0, 'x_max': 1000,
        'y_min': 0, 'y_max': 1000
    }
    TIME_WINDOW_SECONDS = 2  # Only keep last x seconds of data
    
    # Create the mapper
    mapper = TrackMapper(MAP_IMAGE_PATH, COORDINATE_BOUNDS, TIME_WINDOW_SECONDS)
    
    # Create MQTT manager
    manager = MQTTMultiSourceManager(broker_address="mqtt.eclipseprojects.io")
    manager.add_source("all_tracking", "tracking/+/+/+", camera_handler)
    
    # Set enhanced callbacks that include the mapper
    manager.set_callbacks(
        on_track_update=lambda source, track_id, track_data, prev: enhanced_on_track_update(track_data, mapper),
        on_new_track=lambda source, track_id, track_data: enhanced_on_new_track(track_data, mapper),
        on_track_lost=lambda source, track_id, track_data: enhanced_on_track_lost(track_data, mapper)
    )
    
    manager.connect()

    target_str = "Target"
    
    try:
        iteration = 0
        while True:
            if manager.is_connected:
                iteration += 1
                mapper.update_visualization()
                
                # Update visualization every iteration
                if iteration % 1 == 0:
                    #mapper.print_summary()
                    mapper.handle_nao_angle(target=target_str)
                
                
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        manager.disconnect()
        plt.close('all')
        print("Mapper stopped.")