def get_button_style(button_type='default'):
    from config.app_settings import UI_COLORS

    styles = {
        'default': f"""
            QPushButton {{
                min-height: 35px;
                background-color: {UI_COLORS['light']};
                color: {UI_COLORS['dark']};
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px 15px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #e0e0e0;
            }}
            QPushButton:disabled {{
                color: #999999;
                background-color: #e0e0e0;
            }}
        """,
        'primary': f"""
            QPushButton {{
                min-height: 35px;
                background-color: {UI_COLORS['primary']};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 15px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #1565C0;
            }}
            QPushButton:disabled {{
                background-color: #90CAF9;
            }}
        """,
        'success': f"""
            QPushButton {{
                min-height: 35px;
                background-color: {UI_COLORS['success']};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 15px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #388E3C;
            }}
            QPushButton:disabled {{
                background-color: #A5D6A7;
            }}
        """,
        # Add more styles as needed
    }

    return styles.get(button_type, styles['default'])


def get_input_style():
    return """
        QLineEdit {
            min-height: 35px;
            padding: 0 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            background-color: #FAFAFA;
        }
        QLineEdit:focus {
            border: 2px solid #1976D2;
            background-color: white;
        }
    """


def get_label_style(is_header=False):
    if is_header:
        return """
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #424242;
                padding: 5px 0;
            }
        """
    return """
        QLabel {
            font-size: 14px;
            color: #616161;
        }
    """


def get_app_stylesheet():
    from config.app_settings import UI_COLORS

    return f"""
        QWidget {{
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 14px;
            color: {UI_COLORS['text']};
        }}

        QMainWindow, QDialog {{
            background-color: {UI_COLORS['background']};
        }}

        QScrollArea {{
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            background: white;
        }}

        QScrollBar:vertical {{
            border: none;
            background: #f0f0f0;
            width: 10px;
            margin: 0px;
        }}

        QScrollBar::handle:vertical {{
            background: #c0c0c0;
            min-height: 20px;
            border-radius: 5px;
        }}

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}

        QToolTip {{
            border: 1px solid #d0d0d0;
            background-color: white;
            padding: 5px;
            border-radius: 3px;
        }}
    """
