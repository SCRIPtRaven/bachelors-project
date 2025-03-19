from core.model import Model


class DriverModel(Model):
    """
    Model class that manages driver data.
    """

    def __init__(self):
        super().__init__()
        self.drivers = []

    def clear_drivers(self):
        """Clears all drivers from the model."""
        self.drivers = []

    def add_driver(self, driver):
        """Adds a driver to the model."""
        self.drivers.append(driver)

    def get_driver_by_id(self, driver_id):
        """Retrieves a driver by its ID."""
        for driver in self.drivers:
            if driver.id == driver_id:
                return driver
        return None

    def update_driver_stats(self, driver_id, stats):
        """
        Updates a driver's statistics.

        Args:
            driver_id: The ID of the driver to update
            stats: Dictionary of statistics to update
        """
        driver = self.get_driver_by_id(driver_id)
        if driver and hasattr(driver, 'stats'):
            driver.stats.update(stats)
        elif driver:
            driver.stats = stats
